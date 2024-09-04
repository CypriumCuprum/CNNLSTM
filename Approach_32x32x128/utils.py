import torch

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

from torch.utils.data import DataLoader

from dataset_32x32x128 import EEGDataset128Channel, Splitter


def evaluate(model, test_loader, device):
    model.eval()
    torch.set_grad_enabled(False)
    confusion_matrix = torch.zeros(40, 40)
    for i, (signal, label) in enumerate(test_loader):
        signal = signal.to(device)
        label = label.to(device, dtype=torch.int64)

        output = model(signal)
        _, predicted = output.max(1)
        for j in range(len(predicted)):
            confusion_matrix[label[j]][predicted[j]] += 1

    # Plot confusion matrix

    cm = confusion_matrix.numpy()
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=np.arange(40), yticklabels=np.arange(40))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("EEG Signal Classification")

    parser.add_argument("--file_model", type=str, default="CNNModel128", help="Model type")
    parser.add_argument("--filedata", type=str, default="../data/eeg_signals_32x32_128.pth", help="Path to data file")
    parser.add_argument("--splits_path", type=str, default="data/block_splits_by_image_all.pth",
                        help="Path to split file")
    parser.add_argument("--split_num", type=int, default=0, help="Split number")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.file_model)
    EEGData = EEGDataset128Channel(args.filedata, 0, -1)
    loader = {
        "test": DataLoader(Splitter(EEGData, args.splits_path, args.split_num, "test"), batch_size=args.batch_size,
                           shuffle=False)}

    evaluate(model, loader["test"], device)
