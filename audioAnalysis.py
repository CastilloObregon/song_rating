import librosa
import librosa.display
import IPython.display as ipd
import matplotlib as matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import sklearn
import numpy as np
import scipy
import glob
import os
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

from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

features = pd.read_csv(r'dataset.csv')
features = pd.DataFrame(features)
print(features)

# with open('dataset.csv', newline='') as f:
#     reader = csv.reader(f)
#     data = list(reader)

# print(data)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# print(scaled_features)

kmeans = KMeans(
    init="random",
    n_clusters=5,
    n_init=10,
    max_iter=300,
    random_state=42
    )

# ======= Ejecucion de Kmeans clustering ========
kmeans.fit(scaled_features)

print(kmeans.inertia_)
print(kmeans.cluster_centers_)
print(kmeans.n_iter_)

print(kmeans.labels_)

# =========================================== CODIGO AUXILIAR ===========================================

# data = pd.read_csv('dataset.csv')
# data.head()  # Dropping unneccesary columns
# data = data.drop(['filename'], axis=1)  # Encoding the Labels
# genre_list = data.iloc[:, -1]
# encoder = LabelEncoder()
# y = encoder.fit_transform(genre_list)  # Scaling the Feature columns
# scaler = StandardScaler()
# # Dividing data into training and Testing set
# X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = Sequential()
# model.add(layers.Dense(256, activation='relu',
#                        input_shape=(X_train.shape[1],)))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# classifier = model.fit(X_train,
#                        y_train,
#                        epochs=100,
#                        batch_size=128)
