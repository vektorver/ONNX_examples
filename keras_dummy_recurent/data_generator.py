"""
Dummy data generator for dummy recurrent neural network.
Format of data:
    X_train: [data_size, sequence_length]
    y_train: [data_size, num_classes]

"""

import json
import os
import shutil

import numpy as np

TRAIN_DATA_SIZE = 1000
TEST_DATA_SIZE = 10
DATA_PATH = 'data'

X_train = np.random.randint(TRAIN_DATA_SIZE, size=(TRAIN_DATA_SIZE, 100))
X_test = np.random.randint(TEST_DATA_SIZE, size=(TEST_DATA_SIZE, 100))

y_train = np.random.uniform(size=(TRAIN_DATA_SIZE, 10))
y_test = np.random.uniform(size=(TEST_DATA_SIZE, 10))

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)

os.mkdir(DATA_PATH)

# save data as json
with open(os.path.join(DATA_PATH, 'X_train.json'), 'w', encoding='utf-8') as outfile:
    json.dump(X_train.tolist(), outfile)

with open(os.path.join(DATA_PATH, 'y_train.json'), 'w', encoding='utf-8') as outfile:
    json.dump(y_train.tolist(), outfile)

with open(os.path.join(DATA_PATH, 'X_test.json'), 'w', encoding='utf-8') as outfile:
    json.dump(X_test.tolist(), outfile)

with open(os.path.join(DATA_PATH, 'y_test.json'), 'w', encoding='utf-8') as outfile:
    json.dump(y_test.tolist(), outfile)
