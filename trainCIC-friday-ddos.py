import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
import numpy as np
df = pd.read_csv('data/thursday-webattack-preprocess.csv',
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
                                                    test_size=1 / 3,
                                                    shuffle=True)

# The known number of output classes.
num_classes = 4

# Input image dimensions
# input_shape = (4,)

# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_test_binary = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)

num_xrecord = len(x_train)
x_train = x_train.reshape(num_xrecord, 7, 1)
x_test = x_test.reshape(len(x_test), 7, 1)

model = Sequential()
model.add(Conv1D(64, (4), input_shape=(7, 1), activation='tanh'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()
# #------------------------------------
# batch_size = 128
# epochs = 20
# model = model.fit(x_train,
#                   y_train_binary,
#                   batch_size=batch_size,
#                   epochs=epochs,
#                   verbose=1,
#                   validation_data=(x_test, y_test_binary))
