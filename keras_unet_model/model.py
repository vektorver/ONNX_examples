"""Keras implementation of U-Net."""
import json
import os
import shutil
import subprocess
import netron

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

K.clear_session()

TRAIN_DATA_SIZE = 1000
TEST_DATA_SIZE = 10
MODEL_NAME = 'unet'
DATA_PATH = 'data'
RESULT_PATH = 'results'

# remove dirs
if os.path.exists(MODEL_NAME):
    shutil.rmtree(MODEL_NAME)

if os.path.exists(RESULT_PATH):
    shutil.rmtree(RESULT_PATH)

if os.path.exists(f'{MODEL_NAME}.onnx'):
    os.remove(f'{MODEL_NAME}.onnx')

os.mkdir(RESULT_PATH)

with open(os.path.join(DATA_PATH, 'X_test.json'), encoding='utf-8') as f:
    X_test = np.array(json.load(f))


def unet(num_classes=3, input_shape=(88, 120, 3)):
    """U-Net model with batch normalization."""
    img_input = tf.keras.layers.Input(input_shape)

    # Block 1
    data = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name='block1_conv2')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    block_1_out = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.MaxPooling2D()(block_1_out)

    # Block 2
    data = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name='block2_conv1')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name='block2_conv2')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    block_2_out = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.MaxPooling2D()(block_2_out)

    # Block 3
    data = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='block3_conv1')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='block3_conv2')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='block3_conv3')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    block_3_out = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.MaxPooling2D()(block_3_out)

    # Block 4
    data = tf.keras.layers.Conv2D(512, (3, 3), padding='same', name='block4_conv1')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(512, (3, 3), padding='same', name='block4_conv2')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(512, (3, 3), padding='same', name='block4_conv3')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    block_4_out = tf.keras.layers.Activation('relu')(data)
    data = block_4_out

    # UP 2
    data = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.concatenate([data, block_3_out])
    data = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    # UP 3
    data = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.concatenate([data, block_2_out])
    data = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    # UP 4
    data = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.concatenate([data, block_1_out])
    data = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(data)
    data = tf.keras.layers.BatchNormalization()(data)
    data = tf.keras.layers.Activation('relu')(data)

    data = tf.keras.layers.Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(data)

    model = Model(img_input, data)

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics='mse')

    return model


model_unet = unet(num_classes=3, input_shape=(88, 120, 3))
predictions = model_unet.predict(X_test)

# save predictions to results/py.json
with open(os.path.join(RESULT_PATH, 'py.json'), 'w', encoding='utf-8') as f:
    json.dump(predictions.tolist(), f)

tf.keras.models.save_model(model_unet, MODEL_NAME)

# Run the command line.
proc = subprocess.run(f'python -m tf2onnx.convert --saved-model {MODEL_NAME} '
                      f'--output {MODEL_NAME}.onnx --opset 12'.split(), capture_output=True, check=True)
print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))

netron.start(f'{MODEL_NAME}.onnx')
