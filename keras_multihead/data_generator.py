"""
Data generator for keras duumy model with multihead attention layer
x_train.shape = (1000, 4)
y_train.shape = (1000,)
"""
import os
import shutil
import json

import numpy as np

TRAIN_DATA_SIZE = 1000
TEST_DATA_SIZE = 10
DATA_PATH = 'data'

# remove dirs
if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)

os.mkdir(DATA_PATH)

# random data
x_train = np.random.rand(TRAIN_DATA_SIZE, 11, 4)
y_train = np.random.randint(0, 2, size=(TRAIN_DATA_SIZE,4))

x_test = np.random.rand(TEST_DATA_SIZE, 11, 4)
y_test = np.random.randint(0, 2, size=(TEST_DATA_SIZE,4))

# save data
with open(os.path.join(DATA_PATH, 'X_train.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(x_train.tolist()))

with open(os.path.join(DATA_PATH, 'y_train.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(y_train.tolist()))

with open(os.path.join(DATA_PATH, 'X_test.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(x_test.tolist()))

with open(os.path.join(DATA_PATH, 'y_test.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(y_test.tolist()))