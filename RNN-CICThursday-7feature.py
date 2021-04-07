from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed, Input, LSTM, Conv2D, Flatten, Conv1D
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

# evaluate model


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


df = pd.read_csv('data/CICIDS-thursday.csv',
                 header=None,
                 names=[
                     'Destination', ' FlowDuration', ' FwdPacketLengthMean',
                     'FlowBytesPerSec', ' FlowPacketsPerSec', ' FlowIATMean',
                     'FwdPacketsPerSec', 'class'
                 ])

df['labels'] = df['class'].astype('category').cat.codes
X = df[[
    'Destination', ' FlowDuration', ' FwdPacketLengthMean', 'FlowBytesPerSec',
    ' FlowPacketsPerSec', ' FlowIATMean', 'FwdPacketsPerSec'
]]
Y = df['labels']

x_train, x_test, y_train, y_test = train_test_split(np.asarray(X),
                                                    np.asarray(Y),
                                                    test_size=1 / 5,
                                                    shuffle=True)

# The known number of output classes.
num_classes = 4

# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.reshape(x_train.shape[0], 1, 7, 1)
x_test = x_test.reshape(x_test.shape[0], 1, 7, 1)

print("xtrain shape 0: ", x_train.shape[0])

print(x_train[0:2])
# model = Sequential()
# model.add(TimeDistributed(LSTM(128, input_shape=(1, 7, 1))))
# model.add(LSTM(128))
# model.add(Dense(num_classes, activation='softmax'))

# opt = keras.optimizers.Adam(learning_rate=0.01)
# model.compile(optimizer=opt,
#               loss='binary_crossentropy',
#               metrics=['accuracy', f1_m, precision_m, recall_m])

# # Training.
# batch_size = 128
# epochs = 1
# model.fit(x_train,
#           y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_split=0.25)
# print("********************")
# print(model.summary())
# result = model.evaluate(x_test, y_test, batch_size=batch_size)
# print("result: ", result)
# # model.save('5feature-model.h5')

# # loaded = load_model('5feature-model.h5')
# # # Evaluation.
# # loss, accuracy, f1_score, precision, recall = model.evaluate(x_test,
# #                                                              y_test,
# #                                                              verbose=0)
