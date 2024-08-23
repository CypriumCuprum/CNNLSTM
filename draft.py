import os

import matplotlib.pyplot as plt
import torch
from scipy import signal


# read file .pth
def read_pth_file(file_path):
    return torch.load(file_path)


def draw_signal_per_channel(signal, filename="signal.png"):
    # signal is Tensor
    shape_signal = signal.shape
    fig, ax = plt.subplots(shape_signal[0], 1, figsize=(15, 250))
    for i in range(shape_signal[0]):
        ax[i].plot(signal[i].numpy())
    fig.savefig(filename)


def draw_spectrogram(signal_eeg, fs=1000, root="data/spectrogram", filename="spectrogram.png"):
    f, t, Sxx = signal.spectrogram(signal_eeg, fs)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    plt.axis('off')
    filepath = root + "/" + filename
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)


def draw_spectrogram_folder(signal_dataset_128, index=0, root="data/spectrogram"):
    signal_eeg = signal_dataset_128['eeg']
    label = signal_dataset_128['label']
    path_parent = root + "/" + str(label)
    if not os.path.exists(path_parent):
        os.makedirs(path_parent)
    path_parent = path_parent + "/" + str(index)
    if not os.path.exists(path_parent):
        os.makedirs(path_parent)
    for ii in range(128):
        x_eeg = signal_eeg[ii]
        draw_spectrogram(x_eeg, root=path_parent, filename=f"spectrogram_{label}_{index}_{ii}.png")


if __name__ == "__main__":
    filedata = "data/eeg_55_95_std.pth"
    f_content = read_pth_file(filedata)
    # print(f_content)
    # print("Dataset level 0:")
    # print("Key of big dataset:", f_content.keys())
    # print("Length of dataset:", len(f_content['dataset']))
    # print("Type of <labels>:", type(f_content['labels']))
    # print("Length of <labels>:", len(f_content['labels']))
    # print("Labels:", f_content['labels'])
    # print("Shape of <means>:", f_content['means'].shape)
    # print("Shape of <stddevs>:", f_content['stddevs'].shape)
    #
    print("Key of one dataset item:", f_content['dataset'][0].keys())
    # Type of each key in one dataset item
    print("\nDataset Item 0:")
    print("Type of <eeg>:", type(f_content['dataset'][0]['eeg']))
    print("Type of <label>:", type(f_content['dataset'][0]['label']))
    print("Type of <image>:", type(f_content['dataset'][0]['image']))
    print("Type of <subject>:", type(f_content['dataset'][0]['subject']))
    #
    print("Shape of <eeg>", f_content['dataset'][0]['eeg'].shape)
    print("Label: ", f_content['dataset'][0]['label'])
    print("Image: ", f_content['dataset'][0]['image'])
    print("Subject: ", f_content['dataset'][0]['subject'])
    #
    # # print f_content dataset[1]
    # print("\nDataset Item 1:")
    #
    # print("Shape of <eeg>", f_content['dataset'][1]['eeg'].shape)
    # print("Label: ", f_content['dataset'][1]['label'])
    # print("Image: ", f_content['dataset'][1]['image'])
    # print("Subject: ", f_content['dataset'][1]['subject'])
    # for item in f_content['dataset']:
    #     print(item.keys())

    # Draw signal per channel
    # draw_signal_per_channel(f_content['dataset'][0]['eeg'], filename="data/signal_filtered.png")

    from time import time

    start = time()
    for dt in range(10):
        draw_spectrogram_folder(f_content['dataset'][dt], index=dt, root="data/spectrogram")
    print("Time to draw spectrogram: ", time() - start)
    eeg = f_content['dataset']
