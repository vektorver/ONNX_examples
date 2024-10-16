"""
This is an example of pure tensorflow model (without keras)
"""
import os
import shutil
import subprocess

import netron
import numpy as np
import tensorflow as tf

MODEL_NAME = 'dense'
RESULTS_PATH = 'results'

if os.path.exists(MODEL_NAME):
    shutil.rmtree(MODEL_NAME)

if os.path.exists(RESULTS_PATH):
    shutil.rmtree(RESULTS_PATH)

if os.path.exists(f'{MODEL_NAME}.onnx'):
    os.remove(f'{MODEL_NAME}.onnx')

os.mkdir(RESULTS_PATH)


class MyModel(tf.Module):
    """Simple dense model."""

    # input shape 4, output shape 2
    def __init__(self):
        super(MyModel, self).__init__()
        self.weights = tf.Variable(tf.random.normal([4, 2]))
        self.bias = tf.Variable(tf.random.normal([2]))
        self.sigmoid = tf.keras.activations.sigmoid

    @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float32)])
    def __call__(self, inp):
        data = tf.matmul(inp, self.weights) + self.bias
        return self.sigmoid(data)


model = MyModel()

if os.path.exists(MODEL_NAME):
    shutil.rmtree(MODEL_NAME)

os.mkdir(MODEL_NAME)

# print prediction
predictions = model(np.array([[1.7640524, 0.4001572, 0.978738, 2.2408931]]).astype(np.float32)).numpy()

with open(os.path.join(RESULTS_PATH, 'py.json'), 'w', encoding='utf-8') as f:
    f.write(str(predictions.tolist()))

# save model
tf.saved_model.save(model, MODEL_NAME)

# Run the command line.
proc = subprocess.run(f'python -m tf2onnx.convert --saved-model {MODEL_NAME} '
                      f'--output {MODEL_NAME}.onnx --opset 12'.split(), capture_output=True, check=True)
print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))

netron.start(f'{MODEL_NAME}.onnx')
