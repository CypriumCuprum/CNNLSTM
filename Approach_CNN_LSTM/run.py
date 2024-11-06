from dataset import EEGDataset128Channel, Splitter, make_split
from model import *

import torch
import argparse
from trainer import train
import random
import numpy
import os
from time import time, strftime

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser("EEG Signal Classification")

parser.add_argument("--model_type", type=str, default="CNN128Simple", help="Model type")
parser.add_argument("--filedata", type=str, default="../data/eeg_signals_32x32_128.pth", help="Path to data file")
parser.add_argument("--splits_path", type=str, default="data/block_splits_by_image_all.pth", help="Path to split file")
parser.add_argument("--start", type=int, default=0, help="Start index for dataset")
parser.add_argument("--end", type=int, default=-1, help="End index for dataset")
parser.add_argument("--path_new_split", type=str, default="newnew.pth", help="Path to new split, if not exist")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
parser.add_argument("--split_num", type=int, default=0,
                    help="Split number")  # should always set 0 to avoid error when make split if split is not exist
parser.add_argument("--saveCheck", type=int, default=10, help="Save check point")
parser.add_argument("--seed", type=int, default=0, help="Seed for random")
parser.add_argument("--lr_step", type=int, default=10, help="Step for learning rate scheduler")
parser.add_argument("--model_path", type=str, default="", help="Path to save model")
parser.add_argument("--save_dir", type=str, default="history", help="Path to save model")
parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
parser.add_argument("--num_cnn_layers", type=int, default=3, help="Number of CNN layers")
parser.add_argument("--num_lstm_layers", type=int, default=2, help="Number of LSTM layers")
parser.add_argument("--is_resnet", type=bool, default=False, help="Use ResNet")
parser.add_argument("--is_bidirectional", type=bool, default=False, help="Use Bidirectional LSTM")

args = parser.parse_args()

# print parameter in args
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    print(f"Create folder {args.save_dir}")

# get time form hhmmss
# args.save_dir = f"history/{args.model_type}_{strftime('%H%M%S')}"

args.save_dir = f"history/{args.model_type}_{args.num_cnn_layers}_{args.num_lstm_layers}_{args.is_resnet}_{args.is_bidirectional}_{strftime('%H%M%S')}"
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# write arguments to file
with open(os.path.join(args.save_dir, "parameters.txt"), "w") as f:
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")

torch.manual_seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
numpy.random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EEGData = EEGDataset128Channel(args.filedata, args.start, args.end)

if args.path_new_split:
    path_new_split = args.path_new_split
    length_dataset = len(EEGData)
    print(length_dataset)
    ratio = {"train": 70, "val": 15, "test": 15}
    make_split(length_dataset, ratio, path_new_split)
    args.splits_path = args.path_new_split
print(len(EEGData))
print(args.splits_path)
loader = {split: DataLoader(Splitter(EEGData, args.splits_path, args.split_num, split), batch_size=args.batch_size,
                            shuffle=True) for split in ["train", "val", "test"]}

model = CNN_LSTM()
if args.model_type == "CNN_LSTM":
    model = CNN_LSTM()
elif args.model_type == "CNN4L_LSTM":
    model = CNN4L_LSTM()
elif args.model_type == "Flex_Model":
    model = Flex_Model(args.num_cnn_layers, args.num_lstm_layers, args.is_resnet, args.is_bidirectional)
elif args.model_type == "Flex_Model_noFC":
    model = Flex_Model_noFC(args.num_cnn_layers, args.num_lstm_layers, args.is_resnet, args.is_bidirectional)

# if args.model_path:
#     model = torch.load(args.model_path, weights_only=False)

model.to(device)
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

schedule_lr = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

# Train
train(model, loader, optimizer, device, schedule_lr, args)
