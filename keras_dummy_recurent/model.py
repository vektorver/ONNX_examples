"""
Dummy recurrent model for testing ONNX Converter.
"""

import json
import os
import shutil
import subprocess
import netron

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

K.clear_session()

TRAIN_DATA_SIZE = 1000
TEST_DATA_SIZE = 10
MAX_WORDS_COUNT = 10000
DATA_PATH = 'data'
MODEL_NAME = 'dummy_recurrent'
RESULT_PATH = 'results'

# remove dirs
if os.path.exists(MODEL_NAME):
    shutil.rmtree(MODEL_NAME)

if os.path.exists(RESULT_PATH):
    shutil.rmtree(RESULT_PATH)

if os.path.exists(f'{MODEL_NAME}.onnx'):
    os.remove(f'{MODEL_NAME}.onnx')

os.mkdir(RESULT_PATH)

# Read Data
with open(os.path.join(DATA_PATH, 'X_train.json')) as f:
    X_train = np.array(json.load(f))

with open(os.path.join(DATA_PATH, 'y_train.json')) as f:
    y_train = np.array(json.load(f))

with open(os.path.join(DATA_PATH, 'X_test.json')) as f:
    X_test = np.array(json.load(f))

with open(os.path.join(DATA_PATH, 'y_test.json')) as f:
    y_test = np.array(json.load(f))


def create_model(max_words_count: int) -> Model:
    """Create recurrent model."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(max_words_count, 16, input_length=100))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)))
    model.add(tf.keras.layers.UpSampling1D(2))
    model.add(tf.keras.layers.ZeroPadding1D(1))
    model.add(tf.keras.layers.GRU(16, reset_after=True))
    model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    return model


# create model
rec_model = create_model(MAX_WORDS_COUNT)

# train model
rec_model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# make predictions
predictions = rec_model.predict(X_test)

# save predictions to results/py.json
with open(os.path.join(RESULT_PATH, 'py.json'), 'w', encoding='utf-8') as f:
    json.dump(predictions.tolist(), f)

tf.keras.models.save_model(rec_model, MODEL_NAME)

# Run the command line.
proc = subprocess.run(f'python -m tf2onnx.convert --saved-model {MODEL_NAME} '
                      f'--output {MODEL_NAME}.onnx --opset 12'.split(),
                      capture_output=True, check=True)
print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))

netron.start(f'{MODEL_NAME}.onnx')