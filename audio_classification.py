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


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def modelito(audioFeatures, numberLabels, columnas):

    audioFeatures['target'] = np.where(audioFeatures['labelNum'] == 2, 1, 0)
    # audioFeatures['target'] = audioFeatures['label']

    audioFeatures = audioFeatures.drop(
        columns=['filename', 'labelNum'])

    losLabels = audioFeatures['label']
    train, test = train_test_split(audioFeatures, test_size=0.2)
    train, val = train_test_split(train, test_size=0.20)

    # print(len(train), 'train examples')
    # print(len(val), 'validation examples')
    # print(len(test), 'test examples')

    # print(audioFeatures['target'])

    batch_size = 5
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    # print(train_ds.take(1))

    # for feature_batch, label_batch in train_ds.take(1):
    #     print('Every feature:', list(feature_batch.keys()))
    #     print('A batch of files:', feature_batch['chroma_stft'])
    #     print('A batch of targets:', label_batch)

    feature_columns = []
    # print(columnas)
    for header in columnas:
      feature_columns.append(feature_column.numeric_column(header))
    
    # label1 = feature_column.categorical_column_with_vocabulary_list(
    #     'label', audioFeatures.label.unique())   
    # label1_embedding = feature_column.embedding_column(label1, dimension=8)
    # feature_columns.append(label1_embedding)

    
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


    model = tf.keras.Sequential([
      feature_layer,
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dropout(.1),
      layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # model.build()
    # model.summary()

    checkpoint_path = "Entrenamiento/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=40)

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)

    # !ls {checkpoint_dir}
    




def main():
    features = pd.read_csv(r'dataset.csv')
    audioFeatures = pd.DataFrame(features)

    featuresFixed = features.iloc[:, 1:27]
    columnsNames = list(featuresFixed.columns.values)
     
    # print(audioFeatures.head())

    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(audioFeatures[['label']])

    n_clusters = len(label_encoder.classes_)

    # print(true_labels)
    # print("Labels disponibles: ", n_clusters)

    modelito(audioFeatures, true_labels, columnsNames)


if __name__ == "__main__":
    main()





 # audioFeatures.head()# Dropping unneccesary columns
    # audioFeatures = audioFeatures.drop(['filename'],axis=1)#Encoding the Labels

    # rating_list = data.iloc[:, -1]

    # y = label_encoder.fit_transform(rating_list)#Scaling the Feature columns

    # scaler = StandardScaler()

    # X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)