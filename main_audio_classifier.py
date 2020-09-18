import sounddevice as sd
from scipy.io.wavfile import write

import glob
import os

import time

import librosa
import librosa.display
import IPython.display as ipd
import matplotlib as matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import sklearn
import numpy as np
import scipy

from numba import jit, cuda
from numba import vectorize

import pandas as pd

from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import layers
import keras
from keras.models import Sequential
import warnings

from pathlib import Path


def nextAudio():
    
    list_of_files = glob.glob('Testing/Audios_output/*.wav') 
    print(len(list_of_files))
    if((len(list_of_files))>0):
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        fileNumber = latest_file[28:]
        fileNumber = fileNumber[:-4]
    else:
        fileNumber = 0

    return fileNumber


def recordAudio(recNum):
    print(recNum)
    file_number = int(recNum)+1
    fs = 44100  # Sample rate
    seconds = 2  # Duracion de la grabacion

    grabacion = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    print("Antes de grabar el audio asegurese de no hacer ruido antes y despues de grabar.")
    print("Cargando...")
    print("3..")
    time.sleep(1)
    print("2..")
    time.sleep(1)
    print("1..")
    time.sleep(1)
    print("Grabando 15 segundos de audio...")
    sd.wait()  # Wait until recording is finished
    print("Grabacion terminada con exito.")
    write('Testing/Audios_output/output'+str(file_number)+'.wav', fs, grabacion)  # Save as WAV file 
    audio_path = 'Testing/Audios_output/output'+str(file_number)+'.wav'

    return audio_path


def audioFeatures(audioFile):
    data, sr = librosa.load(audioFile, sr=44100)
    dataRms, ind = librosa.effects.trim(data, top_db=25)
    rmsShape = dataRms.shape
    dataShape = data.shape

    return data, sr, dataRms, rmsShape, dataShape, ind


def audioLen(audioName, theData, theSr):
    durAudio = librosa.get_duration(theData, theSr)
    texto = audioName + " dura: "+str(durAudio)
    return texto

def saveWaveplots(audioName, theData, theSr):
    librosa.display.waveplot(theData, sr=theSr)
    filename = 'Testing/Images_output/Waveplots/' + str(audioName) + '.png'
    plt.savefig(filename)
    plt.clf()

    return filename

def melSpectograms(audioName, theData, theSr):

    pianosong, trIndex = librosa.effects.trim(theData)
    # Transformada de Fourier
    # === Short time fourier transform ===
    n_fft = 2048
    hop_length = 512
    D = np.abs(librosa.stft(pianosong, n_fft=n_fft,  hop_length=hop_length))

    # Espectograma de Mel
    DB = librosa.amplitude_to_db(D, ref=np.max)
    librosa.display.specshow(
        DB, sr=theSr, hop_length=hop_length, x_axis='time', y_axis='log')

    filename2 = 'Testing/Images_output/Mel/' + str(audioName) + '.png'
    plt.savefig(filename2)
    plt.clf()

    return filename2

def main():

    fileNumber = nextAudio()

    audio_path = recordAudio(fileNumber)
    
    laData, elSr, dataRms, rmsShape, dataShape, ind = audioFeatures(audio_path)
    audio_duration = audioLen(audio_path, dataRms, elSr)
    print(audio_duration)
    # ========== Trim de audio name
    audio_name = audio_path[:-4]
    audio_name = audio_name[22:]
    
    # ========== Get waveplots
    waveplot_dir = saveWaveplots(audio_name, dataRms, elSr)
    # ========== Get mel spetrograms
    mel_dir = melSpectograms(audio_name, dataRms, elSr)

    print(waveplot_dir, mel_dir)


if __name__ == "__main__":
    main()
