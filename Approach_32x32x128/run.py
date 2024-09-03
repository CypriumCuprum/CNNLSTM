from dataset_32x32x128 import EEGDataset128Channel, Splitter, make_split
from model import CNNModel128

import torch
import argparse
from train import train

from torch.utils.data import DataLoader

parser = argparse.ArgumentParser("EEG Signal Classification")

parser.add_argument("--model_type", type=str, default="CNN128Simple", help="Model type")
parser.add_argument("--filedata", type=str, default="data/eeg_signals_32x32_128.pth", help="Path to data file")
parser.add_argument("--splits_path", type=str, default="data/block_splits_by_image_all.pth", help="Path to split file")
parser.add_argument("--start", type=int, default=0, help="Start index for dataset")
parser.add_argument("--end", type=int, default=10, help="End index for dataset")
parser.add_argument("--path_new_split", type=str, default="", help="Path to new split, if not exist")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
parser.add_argument("--split_num", type=int, default=0,
                    help="Split number")  # should always set 0 to avoid error when make split if split is not exist

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EEGData = EEGDataset128Channel(args.filedata, args.start, args.end)

if args.path_new_split:
    path_new_split = args.path_new_split
    length_dataset = len(EEGData)
    ratio = {"train": 70, "val": 15, "test": 15}
    make_split(length_dataset, ratio, path_new_split)

args.splits_path = args.path_new_split
loader = {split: DataLoader(Splitter(EEGData, args.splits_path, args.split_num, split), batch_size=args.batch_size,
                            shuffle=True) for split in ["train", "val", "test"]}
model = CNNModel128()
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Train
train(model, loader, optimizer, args.epochs, device)
