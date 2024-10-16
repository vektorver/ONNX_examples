"""
Keras model with multihead attention and import to ONNX
"""
import os
import shutil
import json

import netron
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import subprocess

K.clear_session()

TRAIN_DATA_SIZE = 1000
TEST_DATA_SIZE = 10
DATA_PATH = 'data'
MODEL_NAME = 'multihead_attention'
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
with open(os.path.join(DATA_PATH, 'X_train.json'), encoding='utf-8') as f:
    X_train = np.array(json.load(f))

with open(os.path.join(DATA_PATH, 'y_train.json'), encoding='utf-8') as f:
    y_train = np.array(json.load(f))

with open(os.path.join(DATA_PATH, 'X_test.json'), encoding='utf-8') as f:
    X_test = np.array(json.load(f))

with open(os.path.join(DATA_PATH, 'y_test.json'), encoding='utf-8') as f:
    y_test = np.array(json.load(f))

def create_model():

    inputs = tf.keras.layers.Input(shape=(11, 4))
    embed = tf.keras.layers.Embedding(4, 8, input_length=11)(inputs)
    multi_head = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)(embed, embed)
    flatten = tf.keras.layers.Flatten()(multi_head)
    outputs = tf.keras.layers.Dense(4, activation='sigmoid')(flatten)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

model = create_model()

# Train Model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10)

# Test Model
predictions = model.predict(X_test)

with open(os.path.join(RESULT_PATH, 'py.json'), 'w', encoding='utf-8') as f:
    f.write(str(predictions.tolist()))

# Save Model
tf.keras.models.save_model(model, MODEL_NAME)

# Run the command line.
proc = subprocess.run(f'python -m tf2onnx.convert --saved-model {MODEL_NAME} '
                      f'--output {MODEL_NAME}.onnx --opset 13'.split(), capture_output=True, check=True)
print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))

netron.start(MODEL_NAME + '.onnx')
