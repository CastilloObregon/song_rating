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
    print("Somos labels: ", labels)


def modelito(audioFeatures):

    # audioFeatures['target'] = np.where(audioFeatures['labelNum'] == 2, 1, 0)

    audioFeatures = audioFeatures.drop(
        columns=['filename', 'label'])

    print(audioFeatures["labelNum"])

    feature_columns = []
    
    chroma_stft = tf.feature_column.numeric_column("chroma_stft")
    rmse = tf.feature_column.numeric_column("rmse")
    spectral_centroid = tf.feature_column.numeric_column("spectral_centroid")
    spectral_bandwidth = tf.feature_column.numeric_column("spectral_bandwidth")
    zero_crossing_rate = tf.feature_column.numeric_column("zero_crossing_rate")

    mfcc1 = tf.feature_column.numeric_column("mfcc1")
    mfcc2 = tf.feature_column.numeric_column("mfcc2")
    mfcc3 = tf.feature_column.numeric_column("mfcc3")
    mfcc4 = tf.feature_column.numeric_column("mfcc4")
    mfcc5 = tf.feature_column.numeric_column("mfcc5")
    mfcc6 = tf.feature_column.numeric_column("mfcc6")
    mfcc7 = tf.feature_column.numeric_column("mfcc7")
    mfcc8 = tf.feature_column.numeric_column("mfcc8")
    mfcc9 = tf.feature_column.numeric_column("mfcc9")
    mfcc10 = tf.feature_column.numeric_column("mfcc10")
    mfcc11 = tf.feature_column.numeric_column("mfcc11")
    mfcc12 = tf.feature_column.numeric_column("mfcc12")
    mfcc13 = tf.feature_column.numeric_column("mfcc13")
    mfcc14 = tf.feature_column.numeric_column("mfcc14")
    mfcc15 = tf.feature_column.numeric_column("mfcc15")
    mfcc16 = tf.feature_column.numeric_column("mfcc16")
    mfcc17 = tf.feature_column.numeric_column("mfcc17")
    mfcc18 = tf.feature_column.numeric_column("mfcc18")
    mfcc19 = tf.feature_column.numeric_column("mfcc19")
    mfcc20 = tf.feature_column.numeric_column("mfcc20")

    # labelClass = tf.feature_column.categorical_column_with_hash_bucket("label", hash_bucket_size=10)

    feature_columns = [chroma_stft, rmse, spectral_centroid, spectral_bandwidth, zero_crossing_rate, mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7,
    mfcc8, mfcc9, mfcc10, mfcc11, mfcc12, mfcc13, mfcc14, mfcc15, mfcc16,mfcc17,mfcc18, mfcc19, mfcc20]
    
    # audioFeatures = audioFeatures.values.ravel()

    # X_data = audioFeatures.drop("label")
    # losLabels = audioFeatures["label"]

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, losLabels, test_size=0.2)
    # train, val = train_test_split(train, test_size=0.20)

    print(len(X_data), ' all data')
    print(len(X_train), ' train examples')
    print(len(X_test), 'validation examples')
    print(len(Y_train), 'test examples')
    print(len(Y_test), 'test examples')

    # print(audioFeatures['target'])

    """
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

    # print(test_ds)
    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)

    predictions = model.predict(test_ds)
    print("Prediction", predictions)

    # !ls {checkpoint_dir}
    """




def main():
    features = pd.read_csv(r'dataset.csv')
    # features = pd.DataFrame(features)
    # features = features.values.ravel()

    # featuresFixed = features.iloc[:, 1:28]
    # columnsNames = list(featuresFixed.columns.values)
     
    # print(audioFeatures.head())

    # label_encoder = LabelEncoder()
    # true_labels = label_encoder.fit_transform(features[['label']])

    # n_clusters = len(label_encoder.classes_)

    # print(true_labels)
    # print("Labels disponibles: ", n_clusters)

    modelito(features)


if __name__ == "__main__":
    main()





 # audioFeatures.head()# Dropping unneccesary columns
    # audioFeatures = audioFeatures.drop(['filename'],axis=1)#Encoding the Labels

    # rating_list = data.iloc[:, -1]

    # y = label_encoder.fit_transform(rating_list)#Scaling the Feature columns

    # scaler = StandardScaler()

    # X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)