from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed, Input, LSTM, Conv2D, Flatten, Conv1D, Dropout
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization
from sklearn import preprocessing
import keras
import keras_metrics
from keras.models import Sequential, model_from_json, Model, load_model
from keras import backend as K
import numpy as np
import pandas as pd
import pickle
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import tensorflow as tf
import random as python_random
# np.random.seed(1337)
from numpy.random import seed
seed(1)

n = 41


def load_dataset(path):
    global n
    featureNum = []
    for i in range(1, n + 1):
        featureNum.append(str(i))
    featureNum.append('class')
    df = pd.read_csv(path, header=None, names=featureNum)

    df['labels'] = df['class'].astype('category').cat.codes

    df['2'] = df['2'].astype('category').cat.codes
    df['3'] = df['3'].astype('category').cat.codes
    df['4'] = df['4'].astype('category').cat.codes

    X = df[featureNum[0:n]]
    Y = df['labels']

    return np.asarray(X), np.asarray(Y)


path = "data/datamerge/test-41f.csv"
x_test, y_test = load_dataset(path)
numrec = x_test.shape[0]
print(numrec)
num_classes = 2
img_rows, img_cols = 1, n

y_test = keras.utils.to_categorical(y_test, num_classes)

x_test = x_test.reshape(x_test.shape[0], img_cols, 1)

model = load_model('slide-model/cnn-41f.h5')

import time
scale = 10493 / 3029
start = time.time()
model.evaluate(x_test, y_test)
exc = time.time() - start
print("so ban ghi / giây")
print(numrec / exc / scale)
