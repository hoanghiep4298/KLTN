from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed, Input, LSTM, Conv2D, Flatten, Conv1D, Dropout
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, model_from_json, Model, load_model
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from keras.optimizers import SGD
import tensorflow as tf
import random as python_random
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)

# evaluate model

# df = pd.read_csv('data/KDD-train-7feature-A-B-EF-W-X-AF-AG.csv',
#                  header=None,
#                  names=[
#                      'duration', 'protocol_type', 'byte_count', 'count',
#                      ' srv_count', 'dst_host_count', 'dst_host_srv_count',
#                      'class'
#                  ])

# df['labels'] = df['class'].astype('category').cat.codes
# X = df[[
#     'duration', 'protocol_type', 'byte_count', 'count', ' srv_count',
#     'dst_host_count', 'dst_host_srv_count'
# ]]
# Y = df['labels']

# x_train, x_test, y_train, y_test = train_test_split(np.asarray(X),
#                                                     np.asarray(Y),
#                                                     test_size=1 / 3,
#                                                     shuffle=True)

n = 7


def load_dataset(path):
    global n
    featureNum = []
    for i in range(1, n + 1):
        featureNum.append(str(i))
    featureNum.append('class')
    df = pd.read_csv(path, header=None, names=featureNum)

    df['labels'] = df['class'].astype('category').cat.codes

    # df['2'] = df['2'].astype('category').cat.codes
    # df['3'] = df['3'].astype('category').cat.codes
    # df['4'] = df['4'].astype('category').cat.codes

    X = df[featureNum[0:n]]
    Y = df['labels']

    return np.asarray(X), np.asarray(Y)


x_train, y_train = load_dataset('data/datamerge/train-41f.csv')
x_test, y_test = load_dataset('data/datamerge/test-41f.csv')
# x_val, y_val = load_dataset('data/datamerge/val-41f.csv')
# The known number of output classes.
print(x_train[:20])
num_classes = 2
num_feature = 7
# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train = keras.utils.to_categorical(y_train, num_classes)
# y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.reshape(x_train.shape[0], 1, num_feature, 1)
x_test = x_test.reshape(x_test.shape[0], 1, num_feature, 1)
# x_val = x_val.reshape(x_val.shape[0], 1, num_feature, 1)

print(x_train.shape[0])

model = Sequential()
model.add(TimeDistributed(LSTM(128, input_shape=(1, 7, 1))))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(LSTM(128))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Training.
batch_size = 128
epochs = 5
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    # validation_data=(x_val, y_val))
)

result = model.evaluate(x_test, y_test)
print("result: ", result)
print("********************")
print(model.summary())

model.save('slide-model/rnn-7f.h5')

# loaded = load_model('5feature-model.h5')
# # Evaluation.
# loss, accuracy, f1_score, precision, recall = model.evaluate(x_test,
#                                                              y_test,
#                                                              verbose=0)

# print(f1_score)
