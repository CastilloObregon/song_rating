import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

features = pd.read_csv(r'dataset.csv')
dataframe = pd.DataFrame(features)

featuresFixed = features.iloc[:, 0:27]

print(audioFeatures.head())

label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(audioFeatures[['label']])

n_clusters = len(label_encoder.classes_)

print(true_labels)
print("Labels disponibles: ", n_clusters

dataframe['target'] = np.where(dataframe['AdoptionSpeed']==4, 0, 1)

# Drop un-used columns.
dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description']))