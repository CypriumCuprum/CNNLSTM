import unittest

import numpy as np
from matplotlib import pyplot as plt
import json
import pywt
import torch
from multiprocessing.pool import ThreadPool
from time import time
import torch
from tqdm import tqdm


class TestSignalProcessing(unittest.TestCase):
    def test_one_frequency(self):
        signal = np.random.rand(32)
        coef = one_Frequency(signal)
        self.assertIsInstance(coef, np.ndarray)
        self.assertEqual(coef.shape, (32,))

    def test_create32x32(self):
        signal = np.random.rand(500)
        coef3232 = create32x32(signal)
        self.assertIsInstance(coef3232, torch.Tensor)
        self.assertEqual(coef3232.shape, (32, 32))

    def test_make_one_item(self):
        signal128 = np.random.rand(128, 500)
        label = 0
        oneitem = make_one_element(signal128, label)
        self.assertIsInstance(oneitem, dict)
        self.assertIn("eeg", oneitem)
        self.assertIn("label", oneitem)
        self.assertIsInstance(oneitem["eeg"], torch.Tensor)
        self.assertIsInstance(oneitem["label"], torch.Tensor)
        self.assertEqual(oneitem["label"], label)
        self.assertEqual(oneitem["eeg"].shape, (128, 32, 32))

    def test_make_full_dataset(self):
        filedata = "data/eeg_signals_raw_with_mean_std.pth"
        start = 0
        end = 10
        dataset = make_full_dataset(filedata, start, end)
        self.assertIsInstance(dataset, list)
        self.assertEqual(len(dataset), end - start)
        self.assertIsInstance(dataset[0], dict)
        self.assertIn("eeg", dataset[0])
        self.assertIn("label", dataset[0])
        self.assertIsInstance(dataset[0]["eeg"], torch.Tensor)
        self.assertIsInstance(dataset[0]["label"], torch.Tensor)
        self.assertEqual(dataset[0]["eeg"].shape, (128, 32, 32))

    def test_save_processed_dataset(self):
        dataset = [{"eeg": torch.rand(128, 32, 32), "label": torch.tensor(0)}]
        path = "data/eeg_signals_32x32_128_unittest.pth"
        save_processed_dataset(dataset, path)
        dt = torch.load(path)
        self.assertIsInstance(dt, list)
        self.assertEqual(len(dt), 1)
        self.assertIsInstance(dt[0], dict)
        self.assertIn("eeg", dt[0])
        self.assertIn("label", dt[0])
        self.assertIsInstance(dt[0]["eeg"], torch.Tensor)
        self.assertIsInstance(dt[0]["label"], torch.Tensor)
        self.assertEqual(dt[0]["eeg"].shape, (128, 32, 32))


def one_Frequency(signal: np.ndarray):
    coef = pywt.cwt(signal, [1], "mexh")[0]
    return coef[0]


def create32x32(signal: np.ndarray):
    """Creates a 32x32 image from a signal. By apply wavelet transform to each row of the cut signal """
    signal = signal[40:]
    if len(signal) >= 1024:
        signal = signal[:1024]
    signal = np.pad(signal, (0, 1024 - len(signal)), "constant")
    signal = signal.reshape(32, 32)
    coef3232 = np.array([one_Frequency(row) for row in signal])
    coef3232 = torch.from_numpy(coef3232).float()
    return coef3232


def create32x32_type2(signal: np.ndarray):
    """Creates a 32x32 image from a signal. By apply wavelet transform to full signal and then reshape it"""
    signal = signal[40:]
    signal = np.pad(signal, (0, 1024 - len(signal)), "constant")
    coef = one_Frequency(signal)
    coef = np.reshape(coef, (32, 32))
    return coef


def make_one_element(signal128, label):
    signal128_tf = np.array([create32x32(signal128[i]) for i in range(128)])
    signal128_tf = torch.tensor(signal128_tf, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.int32)
    one_item = {'eeg': signal128_tf, 'label': label}
    return one_item


def read_pth_file(filedata):
    f_content = torch.load(filedata)
    return f_content


def make_full_dataset(filedata, start=0, end=-1):
    f_content = read_pth_file(filedata)
    dataset = f_content['dataset']
    if end == -1:
        end = len(dataset)
    dataset_transform = []
    for i in tqdm(range(start, end), desc="Processing dataset: "):
        signal128 = dataset[i]['eeg']
        label = dataset[i]['label']
        one_element = make_one_element(signal128, label)
        dataset_transform.append(one_element)
    return dataset_transform


def save_processed_dataset(dataset, save_path):
    torch.save(dataset, save_path)


if __name__ == "__main__":
    # unittest.main()
    # Load the dataset
    # with open("data/example0.json", "r") as f:
    #     eeg = np.array(json.load(f))

    # Create the 32x32 image
    # create32x32(eeg)
    # one_Frequency(eeg)

    file_path = "data/eeg_signals_raw_with_mean_std.pth"
    start_num = 0
    end_num = 100
    start_time = time()
    processed_dataset = make_full_dataset(file_path, start_num, end_num)
    end_time = time()
    print(f"Time taken: {end_time - start_time}")
    save_path = "data/eeg_signals_32x32_128.pth"
    save_processed_dataset(processed_dataset, save_path)
