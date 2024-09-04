import torch
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter


def train(model, loader, optimizer, device, schedule_lr, args):
    writer = SummaryWriter("runs/" + args.model_type)
    model.train()
    losses_per_epoch = {"train": [], "val": [], "test": []}
    accuracies_per_epoch = {"train": [], "val": [], "test": []}
    best_accuracy = 0
    best_accuracy_val = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        losses = {"train": 0, "val": 0, "test": 0}
        accuracies = {"train": 0, "val": 0, "test": 0}
        counts = {"train": 0, "val": 0, "test": 0}
        for split in ["train", "val", "test"]:
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
                if i % 50 == 0:
                    writer.add_scalars(f"Loss", {split: (losses[split] / counts[split])},
                                       epoch * len(loader[split]) + i)
                    writer.add_scalars(f"Accuracy", {split: (accuracies[split] / counts[split])},
                                       epoch * len(loader[split]) + i)
        # End Epochs
        if accuracies["val"] / counts["val"] >= best_accuracy_val:
            best_accuracy_val = accuracies["val"] / counts["val"]
            best_accuracy = accuracies["test"] / counts["test"]
            best_epoch = epoch

        TrL, TrA, VL, VA, TeL, TeA = losses["train"] / counts["train"], accuracies["train"] / counts["train"], losses[
            "val"] / counts["val"], accuracies["val"] / counts["val"], losses["test"] / counts["test"], accuracies[
                                         "test"] / counts["test"]
        print(
            "Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}, TeA at max VA = {7:.4f} at epoch {8:d}".format(
                epoch, TrL, TrA, VL, VA, TeL, TeA, best_accuracy, best_epoch))

        losses_per_epoch['train'].append(TrL)
        losses_per_epoch['val'].append(VL)
        losses_per_epoch['test'].append(TeL)
        accuracies_per_epoch['train'].append(TrA)
        accuracies_per_epoch['val'].append(VA)
        accuracies_per_epoch['test'].append(TeA)

        if epoch % args.saveCheck == 0:
            if not os.path.exists('rs'):
                os.makedirs('rs')
            torch.save(model, 'rs/%s_epoch_%d.pth' % (args.model_type, epoch))
        schedule_lr.step()
        print(schedule_lr.get_last_lr())
