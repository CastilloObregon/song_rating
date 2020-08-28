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
listaAudios = []
datas = []
srs = []

files = []

noDir = 'Audios/'

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
    # x = np.linspace(0, 2 * np.pi, 400)
    # y = np.sin(x ** 2)
    return data, sr

def listaLimpia():
    for aud in audios:  # iterating on a copy since removing will mess things up
        new_string = aud.replace(noDir, "")
        listaAudios.append(new_string)
    print("Hola somos la lista simplificada: ",listaAudios)
    # print("Hola somos la lista normal: ",audios)
    

# ================================== FUNCIÓN PARA AGREGAR CARACTERÍSTICAS DEL AUDIO A LISTA ===========================

def appenDataAndSr():
    for lesAudios in audios:
        laData, elSr = audioFeatures(lesAudios)
        datas.append(laData)
        srs.append(elSr)
    print("Hola, somos datas: ", datas)
    print("Hola, somos srs: ", srs)

# ================================== FUNCIONES PARA MOSTRAR WAVEPLOTS Y ESPECTOGRAMAS ===========================

def saveWaveplots(audioName, theData, theSr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveplot(theData, sr=theSr)
    # plt.show()
    filename = 'Images/Waveplots/' + str(audioName) +'.png'
    # files.append(filename)
    plt.savefig(filename)

def saveSpectograms(audioName, theData, theSr):
    X = librosa.stft(theData)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=theSr, x_axis='time', y_axis='hz')
    plt.colorbar()
    # plt.show()
    filename = 'Images/Espectogramas/Hz/' + str(audioName) +'.png'
    plt.savefig(filename)
    librosa.display.specshow(Xdb, sr=theSr, x_axis='time', y_axis='log')
    # plt.show()
    filename2 = 'Images/Espectogramas/Log/' + str(audioName) +'.png'
    plt.savefig(filename2)


def melSpectograms(audioName, theData, theSr):

    pianosong, _ = librosa.effects.trim(theData)

    # Transformada de Fourier
    # === Short time fourier transform ===
    n_fft = 2048
    D = np.abs(librosa.stft(pianosong[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
    plt.plot(D)
    # plt.show()
    hop_length = 512
    D = np.abs(librosa.stft(pianosong, n_fft=n_fft,  hop_length=hop_length))
    librosa.display.specshow(D, sr=theSr, x_axis='time', y_axis='linear')
    # plt.colorbar()
    # plt.show()
    filename1 = 'Images/Mel_Spectograms/y_axis_linear/' + str(audioName) +'.png'
    plt.savefig(filename1)

    # Espectograma de Mel
    DB = librosa.amplitude_to_db(D, ref=np.max)
    librosa.display.specshow(
        DB, sr=theSr, hop_length=hop_length, x_axis='time', y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()
    filename2 = 'Images/Mel_Spectograms/y_axis_mel/' + str(audioName) +'.png'
    plt.savefig(filename2)


def main():
    # ========== Preparar la librería de audios ==========
    audioDatabase()
    appenDataAndSr()
    listaLimpia()


    for losAudios, datos, senales in zip(listaAudios, datas, srs):
        saveWaveplots(losAudios, datos,senales)

    for losAudios, datos, senales in zip(listaAudios, datas, srs):
        saveSpectograms(losAudios, datos, senales)

    for losAudios, datos, senales in zip(listaAudios, datas, srs):
        melSpectograms(losAudios, datos, senales)
    
    print(files)

if __name__ == "__main__":
    main()




# ================================== CÓDIGO AUXILIAR ===========================

# ================================== WAVEPLOTS DE MÚLTIPLES AUDIOS, SIMULTÁNEAMENTE ===========================

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