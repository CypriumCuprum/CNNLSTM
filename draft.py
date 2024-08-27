import json
import os
from time import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

matplotlib.use('Agg')


# read file .pth
def read_pth_file(file_path):
    return torch.load(file_path)


def draw_signal_per_channel(signal, index=0, label=0, filename="signal.png", root="data/signal"):
    # signal is Tensor
    shape_signal = signal.shape
    if not os.path.exists(root + "/" + str(label)):
        os.makedirs(root + "/" + str(label))
    filename = f"signal_{index}_{label}.png"
    fig, ax = plt.subplots(shape_signal[0], 1, figsize=(15, 250))
    for i in range(shape_signal[0]):
        ax[i].plot(signal[i].numpy())
    fig.savefig(os.path.join(root, str(label), filename))


def draw_spectrogram(signal_eeg, fs=1000, root="data/spectrogram", filename="spectrogram.png"):
    f, t, Sxx = signal.spectrogram(signal_eeg, fs)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    plt.axis('off')
    filepath = root + "/" + filename
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()


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
    print("dataset[0]['eeg'][0]:")
    with open("data/example0.json", "w") as f:
        json.dump(f_content['dataset'][0]['eeg'][0].tolist(), f)

    # print f_content dataset[1]
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

    start = time()
    # li_one_per_label = [0, 1, 2, 5, 6, 8, 9, 10, 12, 13, 14, 15, 18, 19, 21, 24, 26, 29, 34, 35, 36, 38, 39, 41,
    #                     43, 44, 45, 49, 50, 56, 63, 64, 65, 67, 68, 71, 76, 100, 106, 192]
    li_label_0 = [19, 25, 164, 200, 222, 376, 434, 446, 477, 484, 507, 510, 680, 700]
    # for dt in li_label_0:
    #     draw_spectrogram_folder(f_content['dataset'][dt], index=dt, root="data/spectrogram")
    # print("Time to draw spectrogram: ", time() - start)

    # for dt in li_label_0:
    #     eegsignal = f_content['dataset'][dt]['eeg']
    #     label = f_content['dataset'][dt]['label']
    #     draw_signal_per_channel(eegsignal, index=dt, label=label, root="data/signal")
    # print("Time to draw signal: ", time() - start)

    # Draw spectrogram folder
    # start = time()
    # for dt in range(10):
    #     draw_spectrogram_folder(f_content['dataset'][dt], index=dt, root="data/spectrogram")
    # print("Time to draw spectrogram: ", time() - start)

    # label_check = []
    # li = []
    # for dt in range(len(f_content['dataset'])):
    #     onedata = f_content['dataset'][dt]
    #     label = onedata['label']
    #     if label not in label_check:
    #         label_check.append(label)
    #         print(label)
    #         li.append(dt)
    # print(li)

    # label_0 = []
    # for dt in range(len(f_content['dataset'])):
    #     onedata = f_content['dataset'][dt]
    #     label = onedata['label']
    #     if label == 0:
    #         label_0.append(dt)
    # print(label_0)
