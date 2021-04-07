import numpy as np
from numpy.random import randint as ri
from keras.models import load_model
import time
model = load_model('RNN-final-model.h5')
a = np.array([[[
    ri(0, 57715, 1),
    ri(0, 2, 1),
    ri(0, 1309937401, 1),
    ri(1, 511, 1),
    ri(1, 511, 1),
    ri(1, 255, 1),
    ri(1, 255, 1)
]]])
i = 0
while (i < 20000):
    x = np.array([[[
        ri(0, 57715, 1),
        ri(0, 2, 1),
        ri(0, 1309937401, 1),
        ri(1, 511, 1),
        ri(1, 511, 1),
        ri(1, 255, 1),
        ri(1, 255, 1)
    ]]])
    a = np.append(a, x, axis=0)
    i = i + 1

start = time.time()
result = model.predict(a, verbose=0)
excTime = time.time() - start

# print(timeit.timeit('evaluateSpeed()', number=1))
print(excTime)