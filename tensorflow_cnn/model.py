"""
TensorFlow CNN model
"""
import json
import os
import shutil
import subprocess

import netron
import numpy as np
import tensorflow as tf

TRAIN_DATA_SIZE = 1000
TEST_DATA_SIZE = 10
DATA_PATH = 'data'
MODEL_NAME = 'cnn'
RESULT_PATH = 'results'

if os.path.exists(MODEL_NAME):
    shutil.rmtree(MODEL_NAME)

if os.path.exists(RESULT_PATH):
    shutil.rmtree(RESULT_PATH)

if os.path.exists(f'{MODEL_NAME}.onnx'):
    os.remove(f'{MODEL_NAME}.onnx')

os.mkdir(RESULT_PATH)

# Read Data
with open(os.path.join(DATA_PATH, 'x_train.json'), encoding='utf_8') as f:
    X_train = np.array(json.load(f))

with open(os.path.join(DATA_PATH, 'y_train.json'), encoding='utf_8') as f:
    y_train = np.array(json.load(f))

with open(os.path.join(DATA_PATH, 'x_test.json'), encoding='utf_8') as f:
    X_test = np.array(json.load(f))

with open(os.path.join(DATA_PATH, 'y_test.json'), encoding='utf_8') as f:
    y_test = np.array(json.load(f))

X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


# Define the CNN model
class SimpleCNN(tf.Module):
    """Simple CNN model."""

    def __init__(self, num_classes=10):
        # Convolutional layers
        super(SimpleCNN, self).__init__()
        self.conv1 = tf.Variable(tf.random.normal([3, 3, 3, 32], stddev=0.1))
        self.conv2 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.1))
        # Fully connected layers
        self.fc1_w = tf.Variable(tf.random.normal([4096, 128], stddev=0.1))
        self.fc1_b = tf.Variable(tf.zeros([128]))
        self.fc2_w = tf.Variable(tf.random.normal([128, num_classes], stddev=0.1))
        self.fc2_b = tf.Variable(tf.zeros([num_classes]))

    @tf.function(input_signature=[tf.TensorSpec([None, 32, 32, 3], tf.float32)])
    def __call__(self, inp):
        # Convolutional layers
        data = tf.nn.conv2d(inp, self.conv1, strides=[1, 1, 1, 1], padding='SAME')
        data = tf.nn.relu(data)
        data = tf.nn.max_pool2d(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        data = tf.nn.conv2d(data, self.conv2, strides=[1, 1, 1, 1], padding='SAME')
        data = tf.nn.relu(data)
        data = tf.nn.max_pool2d(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Flatten
        data = tf.reshape(data, shape=[-1, 4096])

        # Fully connected layers
        data = tf.matmul(data, self.fc1_w) + self.fc1_b
        data = tf.nn.relu(data)
        data = tf.matmul(data, self.fc2_w) + self.fc2_b

        return data


# Init model
scnn = SimpleCNN()

# Define optimizer and loss function
optimizer = tf.optimizers.Adam()
loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)

# Training loop
EPOCHS = 10
BATCH_SIZE = 64

for epoch in range(EPOCHS):
    for i in range(0, len(X_train), BATCH_SIZE):
        x_batch, y_batch = X_train[i:i + BATCH_SIZE], y_train[i:i + BATCH_SIZE]

        with tf.GradientTape() as tape:
            logits = scnn(x_batch)
            loss = loss_fn(y_batch, logits)

        gradients = tape.gradient(loss, scnn.variables)
        optimizer.apply_gradients(zip(gradients, scnn.variables))

    # Validation accuracy
    val_logits = scnn(X_test)
    val_accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_test, val_logits))
    print(f'Epoch {epoch + 1}/{EPOCHS}, Validation Accuracy: {val_accuracy.numpy()}')

predictions = scnn(X_test)

# save predictions to results/py.json
with open(os.path.join(RESULT_PATH, 'py.json'), 'w', encoding='utf-8') as f:
    json.dump(predictions.numpy().tolist(), f)

tf.saved_model.save(scnn, MODEL_NAME)

# Run the command line.
proc = subprocess.run(f'python -m tf2onnx.convert --saved-model {MODEL_NAME} '
                      f'--output {MODEL_NAME}.onnx --opset 12'.split(), capture_output=True, check=True)
print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))

netron.start(f'{MODEL_NAME}.onnx')
