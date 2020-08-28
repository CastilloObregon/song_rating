import librosa, librosa.display
import IPython.display as ipd
import matplotlib as matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import sklearn
import numpy as np
import scipy
import glob, os

data, sr = librosa.load('Audios/furelise_11.wav', sr=44100)

# <class 'numpy.ndarray'> <class 'int'>print(x.shape, sr)#(94316,) 22050
print(type(data), type(sr))


print(data.shape)
print(sr)
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)


# mostrar onda sonora
plt.figure(figsize=(14, 5))
librosa.display.waveplot(data, sr=sr)
plt.show()


# mostrar espectograma
X = librosa.stft(data)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.show()

# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.show()


pianosong, _ = librosa.effects.trim(data)
# librosa.display.waveplot(pianosong, sr=sr)
# plt.show()

# Transformada de Fourier
n_fft = 2048
D = np.abs(librosa.stft(pianosong[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
plt.plot(D)
plt.show()


hop_length = 512
D = np.abs(librosa.stft(pianosong, n_fft=n_fft,  hop_length=hop_length))
# librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear');
# plt.colorbar()
# plt.show()

# Espectograma de Mel
DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(
    DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.show()