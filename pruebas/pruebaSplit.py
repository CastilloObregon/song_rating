# ================== SCRIPT FOR MACOS ==================

import librosa, librosa.display
import IPython.display as ipd
import matplotlib as matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import sklearn
import numpy as np
import scipy
import glob, os
from numba import jit, cuda
from numba import vectorize

audios = []
listaAudios = []
datas = []
srs = []

files = []

audiosLen = []

posDatas = []

noDir = 'Audios/'

#os.chdir("Audios")

def audioDatabase():
    for file in glob.glob("Audios/*.wav"):
        audios.append(file)
        # print(file)

    print("Los audios son: ", audios)

def audioFeatures(audioFile):
    data, sr = librosa.load(audioFile, sr=44100)
    dataRms, ind = librosa.effects.trim(data, top_db=2)

    rmsShape = dataRms.shape
    dataShape = data.shape
    #print(type(data), type(sr))
    # print(data)
    # print(sr)
    # x = np.linspace(0, 2 * np.pi, 400)
    # y = np.sin(x ** 2)
    
    return data, sr, dataRms, rmsShape, dataShape, ind

def listaLimpia():
    for aud in audios:  # iterating on a copy since removing will mess things up
        new_string = aud.replace(noDir, "")
        listaAudios.append(new_string)
    print("Hola somos la lista simplificada: ",listaAudios)
    # print("Hola somos la lista normal: ",audios)
    

def appenDataAndSr():
    for lesAudios in audios:
        laData, elSr, dataRms, rmsShape, daShape, splitIndex = audioFeatures(lesAudios)
        
        # laPreData1 = librosa.effects.trim(laData)
        # S = librosa.magphase(librosa.stft(laData, window=np.ones, center=False))[0]
        # laPreData2 = librosa.feature.rms(y=laData)
        
        datas.append(laData)
        posDatas.append(dataRms)
        srs.append(elSr)
        print("Hola somos shape ",rmsShape)
        print("Hola somos shape normal ",daShape)

    print("Hola, somos datas: ", datas)
    print("Hola, somos posDatas: ", posDatas)
    
    print("Hola, somos srs: ", srs)

def audioLen(audioName, theData, theSr):
    durAudio = librosa.get_duration(theData, theSr)
    texto = audioName + " dura: "+str(durAudio)
    audiosLen.append(texto)


def main():
    # ========== Preparar la librería de audios ==========
    audioDatabase()
    appenDataAndSr()
    listaLimpia()

    for losAudios, datos, senales in zip(listaAudios, datas, srs):
        audioLen(losAudios, datos,senales)
    print("La duracion de los audios es: ")
    print(audiosLen)

    for losAudios2, datos2, senales2 in zip(listaAudios, posDatas, srs):
        audioLen(losAudios2, datos2,senales2)
    print("La duracion de los audios es: ")
    print(audiosLen)



if __name__ == "__main__":
    main()
