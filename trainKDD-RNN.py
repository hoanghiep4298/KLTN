from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed, Input, LSTM, Conv2D
from keras.preprocessing import sequence
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json, Model, load_model
from keras import backend as K
import numpy as np
from sklearn.metrics import f1_score

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


df = pd.read_csv('data/trainKDD.csv',
                 header=None,
                 names=[
                     'duration', 'protocol_type', 'byte_count', 'src_count',
                     'dst_host_same_src_port_rate', 'class'
                 ])

df['labels'] = df['class'].astype('category').cat.codes
X = df[[
    'duration',
    'protocol_type',
    'byte_count',
    'src_count',
]]
Y = df['labels']

x_train, x_test, y_train, y_test = train_test_split(np.asarray(X),
                                                    np.asarray(Y),
                                                    test_size=1 / 3,
                                                    shuffle=True)

# The known number of output classes.
num_classes = 2

# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

row_hidden = 128
col_hidden = 128
batch_size = 128
epochs = 1

x_train = x_train.reshape(x_train.shape[0], 1, 4, 1)
x_test = x_test.reshape(x_test.shape[0], 1, 4, 1)

print("xtrain shape 0: ", x_train.shape[0])
# model.summary()
# x = Input(shape=(1, 6, 1))
# encoded_rows = TimeDistributed(LSTM(row_hidden))(x)
# encoded_columns = LSTM(col_hidden)(encoded_rows)
# # Final predictions and model.

# prediction = Dense(num_classes, activation='softmax')(encoded_columns)
# model = Model(x, prediction)

model = Sequential()
model.add(TimeDistributed(LSTM(128, input_shape=(1, 4, 1))))
model.add(LSTM(128))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# Training.
model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
model.summary()

model.save('modeltest.h5')

modelloaded = load_model('modeltest.h5')

# # Evaluation.
# loss, accuracy, f1_score, precision, recall = model.evaluate(x_test,
#                                                              y_test,
#                                                              verbose=0)

# print(f1_score)
