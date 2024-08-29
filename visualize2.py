import json

import numpy as np
import pywt
from matplotlib import pyplot as plt


def _create_scaleogram(signal: np.ndarray) -> np.ndarray:
    """Creates scaleogram for signal, and returns it.

    The resulting scaleogram represents scale in the first dimension, time in
    the second dimension, and the color shows amplitude.
    """
    signal = signal[20:460]
    n = len(signal)  # 128

    # In the PyWavelets implementation, scale 1 corresponds to a wavelet with
    # domain [-8, 8], which means that it covers 17 samples (upper - lower + 1).
    # Scale s corresponds to a wavelet with s*17 samples.
    # The scales in scale_list range from 1 to 16.75. The widest wavelet is
    # 17*16.75 = 284.75 wide, which is just over double the size of the signal.
    scale_list = np.arange(start=0, stop=32) + 1  # 128
    wavelet = "mexh"
    scaleogram = pywt.cwt(signal, scale_list, wavelet)[0]
    return scaleogram


def _create_spectrogram(signal: np.ndarray) -> np.ndarray:
    """
    Creates spectrogram for signal, and returns it.

    The resulting spectrogram represents time on the x axis, frequency
    on the y axis, and the color shows amplitude.
    """
    n = len(signal)  # 128
    sigma = 3
    time_list = np.arange(n)
    spectrogram = np.zeros((n, n))

    for (i, time) in enumerate(time_list):
        # We isolate the original signal at a particular time by multiplying
        # it with a Gaussian filter centered at that time.
        g = _get_gaussian_filter(time, time_list, sigma)
        ug = signal * g
        # Then we calculate the FFT. Some FFT values may be complex, so we take
        # the absolute value to guarantee that they're all real.
        # The FFT is the same size as the original signal.
        ugt = np.abs(np.fft.fftshift(np.fft.fft(ug)))
        # The result becomes a column in the spectrogram.
        spectrogram[:, i] = ugt

    return spectrogram


def _get_gaussian_filter(b: float, b_list: np.ndarray, sigma: float) -> np.ndarray:
    """Returns the values of a Gaussian filter centered at time value b, for all
    time values in b_list, with standard deviation sigma.
    """
    a = 1 / (2 * sigma ** 2)
    return np.exp(-a * (b_list - b) ** 2)


if __name__ == "__main__":
    # Load the dataset
    with open("data/example0.json", "r") as f:
        eeg = np.array(json.load(f))
    print(eeg.shape)
    # Create the scaleogram
    scaleogram = _create_scaleogram(eeg)
    plt.imshow(scaleogram, aspect="auto", cmap="viridis")
    plt.show()
