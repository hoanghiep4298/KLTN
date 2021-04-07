import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, LSTM, Conv2D, Activation, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
import numpy as np
from keras.layers.wrappers import TimeDistributed
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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

df = pd.read_csv('data/trainKDD.csv', header=None, names=['duration', 'protocol_type', 'src_bytes', 'dst_bytes', 'src_count', 'dst_host_same_src_port_rate', 'class'])

df['labels'] =df['class'].astype('category').cat.codes

X = df[['duration', 'protocol_type', 'src_bytes', 'dst_bytes', 'src_count', 'dst_host_same_src_port_rate']]
Y = df['labels']

x_train, x_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(Y), test_size=1/3, shuffle= True)

# The known number of output classes.
num_classes = 2
img_rows, img_cols = 1, 6

# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_test_binary = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)


x_train = x_train.reshape( x_train.shape[0], 1, img_cols, 1)
x_test = x_test.reshape( x_test.shape[0], 1, img_cols, 1)

input_shape = (None, img_cols, 1)
print(input_shape)

model = Sequential()
model.add(TimeDistributed(Conv1D(128, (3), activation='relu', padding='valid'), input_shape=input_shape))

model.add(TimeDistributed(Flatten()))

model.add(TimeDistributed(Dense(64, activation='relu')))
model.add(LSTM(return_sequences=False, units=2 , recurrent_activation='sigmoid'))
model.add(Dense(2))

# time_distributed_merge_layer = Lambda(function=lambda x: K.mean(x, axis=1, keepdims=False))

model.build((None,)+input_shape)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['acc', f1_m, precision_m, recall_m])

# # #----------------------------
batch_size = 50
epochs = 10
model = model.fit(x_train, y_train_binary,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test_binary))