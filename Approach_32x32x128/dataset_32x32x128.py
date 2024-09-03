import unittest

from torch.utils.data import Dataset
import torch
import random
from torchvision import transforms


class Test_MethodDataset(unittest.TestCase):
    def test_EEGDataset128Channel(self):
        filedata = "../data/eeg_signals_32x32_128.pth"
        start = 0
        end = 10
        test_dataset = EEGDataset128Channel(filedata, start, end)
        self.assertIsInstance(test_dataset, Dataset)
        self.assertEqual(len(test_dataset), 10)
        self.assertEqual(len(test_dataset[0]), 2)
        self.assertEqual(test_dataset[0][0].shape, torch.Size([128, 32, 32]))
        self.assertIsInstance(test_dataset[0][1], torch.Tensor)

    def test_Splitter(self):
        dataset = EEGDataset128Channel("../data/eeg_signals_32x32_128.pth", 0, 10)
        split_path = "../data/split_for_unittest.pth"
        split_num = 0
        split_name = "train"
        test_splitter = Splitter(dataset, split_path, split_num, split_name)
        self.assertEqual(len(test_splitter), 6)
        self.assertEqual(len(test_splitter[0]), 2)
        self.assertEqual(test_splitter[0][0].shape, torch.Size([128, 32, 32]))
        self.assertIsInstance(test_splitter[0][1], torch.Tensor)

        split_name = "test"
        test_splitter = Splitter(dataset, split_path, split_num, split_name)
        self.assertEqual(len(test_splitter), 2)
        self.assertEqual(len(test_splitter[0]), 2)
        self.assertEqual(test_splitter[0][0].shape, torch.Size([128, 32, 32]))
        self.assertIsInstance(test_splitter[0][1], torch.Tensor)

        split_name = "val"
        test_splitter = Splitter(dataset, split_path, split_num, split_name)
        self.assertEqual(len(test_splitter), 2)
        self.assertEqual(len(test_splitter[0]), 2)
        self.assertEqual(test_splitter[0][0].shape, torch.Size([128, 32, 32]))
        self.assertIsInstance(test_splitter[0][1], torch.Tensor)


class EEGDataset128Channel(Dataset):
    def __init__(self, filedata, start, end):
        self.filedata = filedata
        self.data = torch.load(filedata)
        if end != -1:
            self.data = self.data[start:end]
        self.mean, self.std = self.compute_mean_std()

    def compute_mean_std(self):
        mean = torch.zeros(128)
        std = torch.zeros(128)
        for i in range(len(self.data)):
            mean += torch.mean(self.data[i]["eeg"], dim=(1, 2))
            std += torch.std(self.data[i]["eeg"], dim=(1, 2))
        mean /= len(self.data)
        std /= len(self.data)
        print(mean.shape)
        return mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        signal = self.data[index]
        label = signal["label"]
        signal = signal["eeg"]  # 128x32x32
        signal = (signal - self.mean[:, None, None]) / self.std[:, None, None]

        return signal, label


class Splitter(Dataset):
    def __init__(self, dataset, split_path, split_num, split_name):
        self.dataset = dataset
        self.split_path = split_path
        self.split_num = split_num
        self.split_name = split_name

        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if i < len(self.dataset)]
        # Compute size
        self.size = len(self.split_idx)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.dataset[self.split_idx[index]]


def make_split(length_dataset, ratio: dict, save_path):
    train_ratio = ratio["train"]
    val_ratio = ratio["val"]
    test_ratio = ratio["test"]
    assert train_ratio + val_ratio + test_ratio == 100  # Check sum of ratio
    num_data = length_dataset
    num_train = int(num_data * train_ratio / 100)
    num_val = int(num_data * val_ratio / 100)
    num_test = num_data - num_train - num_val
    assert num_train + num_val + num_test == num_data
    full_range = list(range(length_dataset))

    # Split data
    random.shuffle(full_range)
    train_idx = full_range[:num_train]
    val_idx = full_range[num_train:num_train + num_val]
    test_idx = full_range[num_train + num_val:]
    split = {"train": train_idx, "val": val_idx, "test": test_idx}
    split = [split]
    format_split = {"splits": split}
    print(format_split)
    torch.save(format_split, save_path)
