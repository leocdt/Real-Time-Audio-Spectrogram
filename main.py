import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

# Audio parameters
samplerRate = 44100
nb_chanel = 1  # Mono input
nb_fft = 4096  # Number of samples in each frame for the Fast Fourier Transform (FFT)
hop_length = int(nb_fft / 4 * 3)  # 0 <= hop_length < number of samples in a frame (nb_fft)
overlap = nb_fft - hop_length
nb_plot_tf = 80  # Number of time frames to plot in the spectrogram

nb_freqs = nb_fft
f_max_idx = 480  # 1 < f_max_idx < nb_freqs 

window = np.hamming(nb_fft)
amp = np.zeros((nb_plot_tf, f_max_idx))

# Plot and video parameters
fps = 1.0
fig, ax = plt.subplots()
image = ax.imshow(amp.T, aspect="auto")
ax.set_xlabel(f"Time frame")
ax.set_ylabel(f"Frequency")
fig.colorbar(image)
vmax, vmin = 1.0, 0.0

def audio_callback(indata, frames, time, status):
    global amp, vmax

    x = np.mean(indata, axis=1)
    amp = np.roll(amp, -1, axis=0)
    amp[-1] = np.sqrt(np.abs(np.fft.rfft(window * x)))[0:f_max_idx]
    if vmax < np.max(amp[-1]):
        vmax = np.max(amp[-1])

def update(frame):
    image.set_clim(vmin, vmax)
    image.set_data(amp.T[::-1])
    plt.title("Real time Spectrogram")

with sd.InputStream(callback=audio_callback, samplerate=samplerRate, blocksize=nb_fft, channels=nb_chanel, dtype=np.float32):
    ani = FuncAnimation(fig, update, interval=10, blit=False)
    plt.show()