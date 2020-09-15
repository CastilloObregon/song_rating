import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf

from tensorflow.python import keras
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import librosa
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
from keras import layers
import keras
from keras.models import Sequential
import warnings


def main():
    features = pd.read_csv(r'dataset.csv')
    audioFeatures = pd.DataFrame(features)

    featuresFixed = features.iloc[:, 1:27]
    columnsNames = list(featuresFixed.columns.values)
     
    # print(audioFeatures.head())

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(audioFeatures[['label']])

    scaler = StandardScaler()

    X = scaler.fit_transform(np.array(features.iloc[:, 1:-2], dtype = float))

    n_clusters = len(label_encoder.classes_)

    # print(y)
    # print("Labels disponibles: ", n_clusters)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    classifier = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=128)



if __name__ == "__main__":
    main()



