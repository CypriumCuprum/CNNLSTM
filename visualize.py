# import matplotlib.pyplot as plt
# import numpy as np
# import pywt
#
# # Example 1
# x = np.arange(512)
# y = np.sin(2 * np.pi * x / 32)
# coef, freqs = pywt.cwt(y, np.arange(1, 129), 'gaus1')
# plt.matshow(coef)
# plt.show()
#
# # Example 2
# t = np.linspace(-1, 1, 200, endpoint=False)
# sig = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7 * (t - 0.4) ** 2) * np.exp(1j * 2 * np.pi * 2 * (t - 0.4)))
# widths = np.arange(1, 31)
# cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
# plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
#            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
# plt.show()
# -------------------------------------------------
# import matplotlib.pyplot as plt
# import numpy as np
#
# import pywt
#
#
# def gaussian(x, x0, sigma):
#     return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)
#
#
# def make_chirp(t, t0, a):
#     frequency = (a * (t + t0)) ** 2
#     chirp = np.sin(2 * np.pi * frequency * t)
#     return chirp, frequency
#
#
# # generate signal
# time = np.linspace(0, 1, 2000)
# chirp1, frequency1 = make_chirp(time, 0.2, 9)
# chirp2, frequency2 = make_chirp(time, 0.1, 5)
# chirp = chirp1 + 0.6 * chirp2
# chirp *= gaussian(time, 0.5, 0.2)
#
# # plot signal
# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(time, chirp)
# axs[1].plot(time, frequency1)
# axs[1].plot(time, frequency2)
# axs[1].set_yscale("log")
# axs[1].set_xlabel("Time (s)")
# axs[0].set_ylabel("Signal")
# axs[1].set_ylabel("True frequency (Hz)")
# plt.suptitle("Input signal")
#
# # perform CWT
# wavelet = "cmor1.5-1.0"
# # logarithmic scale for scales, as suggested by Torrence & Compo:
# widths = np.geomspace(1, 1024, num=100)
# sampling_period = np.diff(time).mean()
# cwtmatr, freqs = pywt.cwt(chirp, widths, wavelet, sampling_period=sampling_period)
# # absolute take absolute value of complex result
# cwtmatr = np.abs(cwtmatr[:-1, :-1])
#
# # plot result using matplotlib's pcolormesh (image with annoted axes)
# fig, axs = plt.subplots(2, 1)
# pcm = axs[0].pcolormesh(time, freqs, cwtmatr)
# axs[0].set_yscale("log")
# axs[0].set_xlabel("Time (s)")
# axs[0].set_ylabel("Frequency (Hz)")
# axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
# fig.colorbar(pcm, ax=axs[0])
#
# # plot fourier transform for comparison
# from numpy.fft import rfft, rfftfreq
#
# yf = rfft(chirp)
# xf = rfftfreq(len(chirp), sampling_period)
# plt.semilogx(xf, np.abs(yf))
# axs[1].set_xlabel("Frequency (Hz)")
# axs[1].set_title("Fourier Transform")
# plt.tight_layout()
#
# plt.show()
# -------------------------------------------------

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import signal
#
# rng = np.random.default_rng()
#
# fs = 10e3
# N = 1e5
# amp = 2 * np.sqrt(2)
# noise_power = 0.01 * fs / 2
# time = np.arange(N) / float(fs)
# mod = 500 * np.cos(2 * np.pi * 0.25 * time)
# carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
# noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
# noise *= np.exp(-time / 5)
# x = carrier + noise
# print("X: ", x)
# print("Shape of X: ", x.shape)
# f, t, Sxx = signal.spectrogram(x, fs)
# plt.pcolormesh(t, f, Sxx, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
# -------------------------------------------------

# import matplotlib.pyplot as plt
# import numpy as np
# import pywt
#
# x = np.arange(512)
# y = np.sin(2 * np.pi * x / 32)
# coef, freqs = pywt.cwt(y, np.arange(1, 129), 'mexh')
# plt.matshow(coef)
# plt.show()
# -------------------------------------------------
# import pywt
#
# t = np.linspace(-1, 1, 200, endpoint=False)
# sig = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7 * (t - 0.4) ** 2) * np.exp(1j * 2 * np.pi * 2 * (t - 0.4)))
# widths = np.arange(1, 31)
# cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
# plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
#            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
# plt.show()
# -------------------------------------------------
# import matplotlib.pyplot as plt
# import numpy as np
# import pywt
#
# with open("data/example0.json", "r") as file:
#     y = json.load(file)
# coef, freqs = pywt.cwt(y[40:500], np.arange(1, 460), 'mexh')
# plt.matshow(coef)
# plt.savefig("data/spectrogram_example_0.png", bbox_inches='tight', pad_inches=0)
# plt.show()

# import json
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pywt
#
# # Load the data from the JSON file
# with open("data/example0.json", "r") as file:
#     y = json.load(file)
#
# # Perform the continuous wavelet transform
# coef, freqs = pywt.cwt(y[40:500], np.arange(1, 33), 'mexh')
# # coef = pywt.wavedec(y[40:460], 'mexh', level=32)
# # Create the plot
# plt.matshow(coef, aspect='auto')
# plt.axis('off')  # Turn off the axes for a cleaner image
#
# # Save the figure with the desired size
# plt.savefig("data/spectrogram_example_0.png", bbox_inches='tight', pad_inches=0, dpi=32)
# plt.show()
# -------------------------------------------------

import json

import matplotlib.pyplot as plt
import numpy as np
import pywt

# Load the data from the JSON file
with open("data/example0.json", "r") as file:
    y = json.load(file)

# Perform the continuous wavelet transform
coef, freqs = pywt.cwt(y[40:500], np.arange(1, 33), 'mexh', sampling_period=1)

# Create a figure with specific size (1x1 inch) and DPI (32)
# fig, ax = plt.subplots(figsize=(1, 1), dpi=42)
fig, ax = plt.subplots()

# Plot the coefficients matrix as a spectrogram
ax.matshow(coef)
ax.axis('off')  # Turn off the axes for a cleaner image

# Save the figure with the desired size and DPI
fig.savefig("data/spectrogram_example_0.png", bbox_inches='tight', pad_inches=0)

# Display the plot
plt.show()

# Close the figure to release resources
plt.close(fig)
