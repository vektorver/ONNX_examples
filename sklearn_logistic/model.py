"""
This is a simple example of how to use the `skl2onnx` package to convert a scikit-learn model to ONNX format.
"""
import os
import shutil
import json

import netron
import numpy as np
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

import onnx
from onnx import helper
from onnx import TensorProto

MODEL_NAME = 'logistic'
RESULT_PATH = 'results'
DATA_PATH = 'data'

# remove dirs
if os.path.exists(MODEL_NAME):
    shutil.rmtree(MODEL_NAME)

if os.path.exists(RESULT_PATH):
    shutil.rmtree(RESULT_PATH)

if os.path.exists(f'{MODEL_NAME}.onnx'):
    os.remove(f'{MODEL_NAME}.onnx')

os.mkdir(RESULT_PATH)

# load data
with open(os.path.join(DATA_PATH, 'X_train.json'), 'r', encoding='utf-8') as f:
    x_train = np.array(json.load(f))

with open(os.path.join(DATA_PATH, 'y_train.json'), 'r', encoding='utf-8') as f:
    y_train = np.array(json.load(f))

with open(os.path.join(DATA_PATH, 'X_test.json'), 'r', encoding='utf-8') as f:
    x_test = np.array(json.load(f))

with open(os.path.join(DATA_PATH, 'y_test.json'), 'r', encoding='utf-8') as f:
    y_test = np.array(json.load(f))

# train model
model = LogisticRegression()
model.fit(x_train, y_train)

# test model
y_pred = model.predict(x_test)
print(f'Accuracy: {np.mean(y_pred == y_test)}')

#save predictions
predictions = model.predict(x_test)

with open(os.path.join(RESULT_PATH, 'py.json'), 'w', encoding='utf-8') as f:
    f.write(str(predictions.tolist()))

# convert model to onnx
initial_type = [('float_input', FloatTensorType([None, x_train.shape[1]]))]

onx = convert_sklearn(model, initial_types=initial_type)

# this model have only one input
# print(onx.graph.input)
# we should create several inputs, one for each feature
# all inputs will be concatenated
# after that, we can connect it to the model

nodes = [
    helper.make_node(
        'Identity',
        inputs=['input_{}'.format(i)],
        outputs=['output_{}'.format(i)],
    ) for i in range(x_train.shape[1])
]

# Create a node that merges the outputs of the nodes into one input
concat_node = helper.make_node(
    'Concat',
    inputs=['output_{}'.format(i) for i in range(x_train.shape[1])],
    outputs=['float_input'],
    axis=1
)


# Create a graph
graph_def = helper.make_graph(
    nodes + [concat_node],
    'IdentityTransform',
    [helper.make_tensor_value_info('input_{}'.format(i), TensorProto.FLOAT, [None, 1]) for i in range(x_train.shape[1])], # Входы графа
    [onx.graph.output[0]],
    )

# Add the graph to the model
graph_def.node.extend(onx.graph.node)

# Create a new model
onx = helper.make_model(graph_def, producer_name='onnx-example')

# set opset 19
onx.opset_import[0].version = 19

# Save the model
onnx.save_model(onx, f'{MODEL_NAME}.onnx')

netron.start(f'{MODEL_NAME}.onnx')

# test the model
sess = rt.InferenceSession(f'{MODEL_NAME}.onnx')
input_names = [inp.name for inp in sess.get_inputs()]
label_names = [out.name for out in sess.get_outputs()]
print(f'Result input names: {input_names}. Output names: {label_names}')
input = {input_names[i]: x_test[:, i].reshape(-1, 1).astype(np.float32) for i in range(x_test.shape[1])}

print(f'ONNX prediction: {sess.run([label_names[0]], input)[0]}')
print(f'Sklearn prediction: {model.predict(x_test)}')