import argparse

import torch
from torch.utils.data import DataLoader

from dataset import EEGDataset128Channel, Splitter
from model import CNN_LSTM, CNN4L_LSTM, Flex_Model, Flex_Model_noFC
from time import time

parser = argparse.ArgumentParser(description="Test model")

parser.add_argument("--model_type", type=str, default="CNN_LSTM", help="Model type")
parser.add_argument("--filedata", type=str, default="../data/eeg_signals_32x32_128.pth", help="Path to data file")
parser.add_argument("--splits_path", type=str, default="data/block_splits_by_image_all.pth", help="Path to split file")
parser.add_argument("--start", type=int, default=0, help="Start index for dataset")
parser.add_argument("--end", type=int, default=-1, help="End index for dataset")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--split_num", type=int, default=0,
                    help="Split number")  # should always set 0 to avoid error when make split if split is not exist
parser.add_argument("--model_path", type=str, default="", help="Path to save model")
parser.add_argument("--save_file", type=str, default="runtime_inference.txt", help="Path to save model")
parser.add_argument("--num_cnn_layers", type=int, default=3, help="Number of CNN layers")
parser.add_argument("--num_lstm_layers", type=int, default=2, help="Number of LSTM layers")
parser.add_argument("--is_resnet", type=bool, default=False, help="Use ResNet")
parser.add_argument("--is_bidirectional", type=bool, default=False, help="Use Bidirectional LSTM")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EEGData = EEGDataset128Channel(args.filedata, args.start, args.end)

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

correct = 0
total = 0

checkpoint = torch.load(args.model_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()
with torch.no_grad():
    for i, (signal, label) in enumerate(loader["test"]):
        signal = signal.to(device)
        label = label.to(device, dtype=torch.int64)

        # Forward
        output = model(signal)

        # Accuracy
        _, predicted = output.max(1)
        correct += predicted.eq(label.data).sum().item()
        total += label.size(0)

print(f"Accuracy Test: {100 * correct / total}")

# print runtime inference for one instance
start_time = time()
model(signal[0:1])
end_time = time()
print(f"Runtime inference for one instance: {end_time - start_time}")

# print continue to file
with open(args.save_file, "a") as f:
    f.write(
        f"{args.model_type} cnn_layer: {args.num_cnn_layers} lstm: {args.num_lstm_layers} is_resnet: {args.is_resnet} is_bidirectional: {args.is_bidirectional} | Runtime inference:{end_time - start_time} | Accuracy Test: {100 * correct / total}\n")

# sample run: python on_Testset.py --model_type CNN_LSTM --model_path history/CNN_LSTM_0.001_4_10_10/checkpoint/checkpoint_best.pth
