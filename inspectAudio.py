import librosa, librosa.display
import IPython.display as ipd
import matplotlib as matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import sklearn
import numpy as np
import scipy
import glob, os

audios = []

datas = []
srs = []

#os.chdir("Audios")
def audioDatabase():
    for file in glob.glob("Audios/*.wav"):
        audios.append(file)
        print(file)

    print("Los audios son: ", audios)



# ================================== FUNCIÓN PARA EXTRAER CARACTERÍSTICAS DEL AUDIO ===========================

def audioFeatures(audioFile):
    data, sr = librosa.load(audioFile, sr=44100)
    #print(type(data), type(sr))
    print(data)
    print(sr)
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2)
    return data, sr

# ================================== FUNCIÓN PARA AGREGAR CARACTERÍSTICAS DEL AUDIO A LISTA ===========================

def appenDataAndSr():
    for lesAudios in audios:
        laData, elSr = audioFeatures(lesAudios)
        datas.append(laData)
        srs.append(elSr)
    print("Hola, somos datas: ", datas)
    print("Hola, somos srs: ", srs)

# ================================== FUNCIONES PARA MOSTRAR WAVEPLOTS Y ESPECTOGRAMAS ===========================

def mostrarWaveplots(theData, theSr):
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(theData, sr=theSr)
    plt.show()


def main():
    # ========== Preparar la librería de audios ==========
    audioDatabase()
    appenDataAndSr()
    
    for datos, senales in zip(datas, srs):
        mostrarWaveplots(datos,senales)

if __name__ == "__main__":
    main()





"""


data, sr = librosa.load('Audios/furelise_01.mp3', sr=44100)

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


"""

# ================================== CÓDIGO AUXILIAR ===========================

# ================================== ESPECTOGRAMAS DE MÚLTIPLES AUDIOS, SIMULTÁNEAMENTE ===========================

# piano_signals = [
#     librosa.load(p.absolute())[0] for p in Path('.').glob('Audios/furelise_*.wav')
# ]

# cuantosAudios = len(piano_signals)
# print("Hay: ",cuantosAudios, " audios")


# plt.figure(figsize=(15, 6))
# for i, x in enumerate(piano_signals):
#     plt.subplot(2, 2, i+1)
#     librosa.display.waveplot(x[:1000000])
    

# plt.show()