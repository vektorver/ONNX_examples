"""
This file is used to generate data for tensorflow_cnn example.

X_train: [TRAIN_DATA_SIZE, WEIGHT, HEIGHT, CHANNEL]
Y_train: [TRAIN_DATA_SIZE, NUM_CLASSES]
"""
import json
import os
import shutil

import numpy as np
import tensorflow as tf

TRAIN_DATA_SIZE = 1000
TEST_DATA_SIZE = 10
DATA_PATH = 'data'

# generate the same data, using random
x_train = np.random.randn(TRAIN_DATA_SIZE, 32, 32, 3).astype(np.float32)
x_test = np.random.randn(TEST_DATA_SIZE, 32, 32, 3).astype(np.float32)
y_train = np.random.randint(0, 10, size=(TRAIN_DATA_SIZE, 1))
y_test = np.random.randint(0, 10, size=(TEST_DATA_SIZE, 1))
# categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# save data
if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)

os.mkdir(DATA_PATH)

# save as json
with open(os.path.join(DATA_PATH, 'x_train.json'), 'w', encoding='utf-8') as outfile:
    json.dump(x_train.tolist(), outfile)

with open(os.path.join(DATA_PATH, 'y_train.json'), 'w', encoding='utf-8') as outfile:
    json.dump(y_train.tolist(), outfile)

with open(os.path.join(DATA_PATH, 'x_test.json'), 'w', encoding='utf-8') as outfile:
    json.dump(x_test.tolist(), outfile)

with open(os.path.join(DATA_PATH, 'y_test.json'), 'w', encoding='utf-8') as outfile:
    json.dump(y_test.tolist(), outfile)
