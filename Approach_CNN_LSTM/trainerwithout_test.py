import torch
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter


def train(model, loader, optimizer, device, schedule_lr, args):
    writer = SummaryWriter(os.path.join(args.save_dir, "runs"))
    model.train()
    losses_per_epoch = {"train": [], "val": []}
    accuracies_per_epoch = {"train": [], "val": []}
    best_accuracy = 0
    best_accuracy_val = 0
    best_epoch = 0
    start_epoch = 0

    if args.model_path:
        checkpoint = torch.load(args.model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['accuracy']

    for epoch in range(start_epoch, args.epochs):
        losses = {"train": 0, "val": 0, "test": 0}
        accuracies = {"train": 0, "val": 0, "test": 0}
        counts = {"train": 0, "val": 0, "test": 0}
        for split in ["train", "val"]:
            if split == "train":
                model.train()
                torch.set_grad_enabled(True)
            else:
                model.eval()
                torch.set_grad_enabled(False)

            for i, (signal, label) in enumerate(loader[split]):
                signal = signal.to(device)
                label = label.to(device, dtype=torch.int64)

                # print("Signal:", signal)
                # Forward
                output = model(signal)

                # Loss
                # print("Output:", output)
                # print("Label:", label)
                loss = F.cross_entropy(output, label)

                # print(loss)
                losses[split] += loss.item()
                # print(losses[split])
                # Accuracy
                _, predicted = output.max(1)
                correct = predicted.eq(label.data).sum().item()
                accuracy = correct / label.size(0)
                accuracies[split] += accuracy
                counts[split] += 1

                # Backward and optimization
                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Tensorboard
                # if i % 50 == 0:
                #     writer.add_scalars(f"Loss", {split: (losses[split] / counts[split])},
                #                        epoch * len(loader[split]) + i)
                #     writer.add_scalars(f"Accuracy", {split: (accuracies[split] / counts[split])},
                #                        epoch * len(loader[split]) + i)
        # End Epochs
        if accuracies["val"] / counts["val"] >= best_accuracy_val:
            best_accuracy_val = accuracies["val"] / counts["val"]
            checkpoint_best = os.path.join(args.save_dir, 'checkpoint', 'checkpoint_best.pth')
            if not os.path.exists(os.path.join(args.save_dir, 'checkpoint')):
                os.makedirs(os.path.join(args.save_dir, 'checkpoint'))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': losses["train"] / counts["train"],
                'accuracy': accuracies["train"] / counts["train"]
            }, checkpoint_best)
            # torch.save(model, 'rs/%s_best.pth' % args.model_type)
            best_epoch = epoch

        TrL, TrA, VL, VA = losses["train"] / counts["train"], accuracies["train"] / counts["train"], losses[
            "val"] / counts["val"], accuracies["val"] / counts["val"]
        print(
            "Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, Best at max VA = {5:.4f} at epoch {6:d}".format(
                epoch, TrL, TrA, VL, VA, best_accuracy_val, best_epoch))

        losses_per_epoch['train'].append(TrL)
        losses_per_epoch['val'].append(VL)
        accuracies_per_epoch['train'].append(TrA)
        accuracies_per_epoch['val'].append(VA)

        writer.add_scalars("Loss", {"train": TrL, "val": VL}, epoch)
        writer.add_scalars("Accuracy", {"train": TrA, "val": VA}, epoch)

        if epoch % args.saveCheck == 0:
            checkpoint_path = os.path.join(args.save_dir, 'checkpoint', 'checkpoint_epoch_%d.pth' % epoch)
            if not os.path.exists(os.path.join(args.save_dir, 'checkpoint')):
                os.makedirs(os.path.join(args.save_dir, 'checkpoint'))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': losses["train"] / counts["train"],
                'accuracy': accuracies["train"] / counts["train"]
            }, checkpoint_path)
            # torch.save(model, 'rs/%s_epoch_%d.pth' % (args.model_type, epoch))
        schedule_lr.step()
        print(schedule_lr.get_last_lr())
