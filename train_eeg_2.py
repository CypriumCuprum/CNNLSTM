# import os
# import warnings
# from time import sleep
#
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
#
# from dataset import EEG_ImageDataset
# from model2 import CNN128Channels
#
# warnings.filterwarnings("ignore")
#
# import argparse
#
# parser = argparse.ArgumentParser(description="Train EEG model")
# parser.add_argument("--numclass", type=int, default=40, help="Number of classes")
# parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
# parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
# parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
# parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
# parser.add_argument("--model", type=str, default="CNN128Channels", help="Model name")
# parser.add_argument("--save_path", type=str, default="save", help="Save path")
# parser.add_argument("--device", type=str, default="cuda", help="Device")
# parser.add_argument("--eeg-dataset", type=str, default="data/spectrogram", help="EEG dataset path")
# args = parser.parse_args()
#
# device = ("cuda" if torch.cuda.is_available() else "cpu")
#
# train_set = EEG_ImageDataset(args.eeg_dataset, numclass=args.numclass)
# valid_set = EEG_ImageDataset(args.eeg_dataset, numclass=args.numclass)
#
# train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)
# valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)
# model = CNN128Channels(num_classes=args.numclass).to(device)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#
# model.train()
#
# num_epochs = 10
#
# best_val_acc = 0
#
# for epoch in range(num_epochs):
#     if epoch % 5 == 0:
#         loop = tqdm(train_loader, total=len(train_loader) + len(valid_loader), position=0, leave=False)
#     else:
#         loop = tqdm(train_loader, total=len(train_loader), position=0, leave=False)
#
#     for x, y in train_loader:
#         loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Training")
#
#         x = x.to(device=device)
#         y = y.to(device=device)
#
#         # print(x.shape)
#         outputs = model(x)
#
#         loss = criterion(outputs, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         loop.set_postfix(train_loss=loss.item())
#         loop.update(1)
#         sleep(0.1)
#
#     if epoch % 5 == 0:
#         num_correct = 0
#         num_samples = 0
#         model.eval()
#
#         with torch.no_grad():
#             for x, y in valid_loader:
#                 loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Validing")
#                 x = x.to(device=device)
#                 y = y.to(device=device)
#
#                 pred = model(x)
#                 pred_idx = torch.argmax(pred, dim=1)
#
#                 num_correct += (pred_idx == y).sum().item()
#                 num_samples += y.size(0)
#
#                 val_acc = round(num_correct / num_samples, 3)
#                 print(val_acc)
#
#                 if val_acc > best_val_acc:
#                     best_val_acc = val_acc
#                     if not os.path.exists("save"):
#                         os.makedirs("save")
#                     torch.save(model.state_dict(), "save/model_best.pth")
#
#                 loop.set_postfix(val_accuracy=val_acc)
#                 loop.update(1)
#                 sleep(0.1)
#
#         model.train()

import os
import warnings
from time import sleep

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter from torch.utils.tensorboard
from tqdm import tqdm

from dataset import EEG_ImageDataset
from model2 import CNN128Channels

warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description="Train EEG model")
parser.add_argument("--numclass", type=int, default=40, help="Number of classes")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--model", type=str, default="CNN128Channels", help="Model name")
parser.add_argument("--save_path", type=str, default="save", help="Save path")
parser.add_argument("--device", type=str, default="cuda", help="Device")
parser.add_argument("--eeg-dataset", type=str, default="data/spectrogram", help="EEG dataset path")
args = parser.parse_args()

device = ("cuda" if torch.cuda.is_available() else "cpu")

train_set = EEG_ImageDataset(args.eeg_dataset, numclass=args.numclass)
valid_set = EEG_ImageDataset(args.eeg_dataset, numclass=args.numclass)

train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)
model = CNN128Channels(num_classes=args.numclass).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.train()

num_epochs = args.num_epochs

best_val_acc = 0

# Initialize TensorBoard writer
if not os.path.exists('save/experiment_1'):
    os.makedirs('save/experiment_1')
writer = SummaryWriter(log_dir='save/experiment_1')

for epoch in range(num_epochs):
    if epoch % 5 == 0:
        loop = tqdm(train_loader, total=len(train_loader) + len(valid_loader), position=0, leave=False)
    else:
        loop = tqdm(train_loader, total=len(train_loader), position=0, leave=False)

    epoch_loss = 0
    for x, y in train_loader:
        loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Training")

        x = x.to(device=device)
        y = y.to(device=device)

        outputs = model(x)

        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        loop.set_postfix(train_loss=loss.item())
        loop.update(1)
        sleep(0.1)

    # Log training loss for this epoch
    writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)

    if epoch % 5 == 0:
        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in valid_loader:
                loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Validating")
                x = x.to(device=device)
                y = y.to(device=device)

                pred = model(x)
                pred_idx = torch.argmax(pred, dim=1)

                num_correct += (pred_idx == y).sum().item()
                num_samples += y.size(0)

                val_acc = round(num_correct / num_samples, 3)
                print(val_acc)

                # Log validation accuracy for this epoch
                writer.add_scalar('Accuracy/val', val_acc, epoch)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if not os.path.exists(args.save_path):
                        os.makedirs(args.save_path)
                    torch.save(model.state_dict(), os.path.join(args.save_path, "model_best.pth"))

                loop.set_postfix(val_accuracy=val_acc)
                loop.update(1)
                sleep(0.1)

        model.train()

# Close the TensorBoard writer
writer.close()
