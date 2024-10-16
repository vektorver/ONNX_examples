"""
This example shows how to export a simple dense layer in PyTorch and convert it to ONNX.
"""
import os
import shutil

import netron
import numpy as np
import torch

FOLDER_PATH = 'dense'
RESULT_PATH = 'results'

if os.path.exists(FOLDER_PATH):
    try:
        shutil.rmtree(FOLDER_PATH)
    except:
        os.remove(FOLDER_PATH)

if os.path.exists(RESULT_PATH):
    shutil.rmtree(RESULT_PATH)

if os.path.exists(f'{FOLDER_PATH}.onnx'):
    os.remove(f'{FOLDER_PATH}.onnx')

os.mkdir(RESULT_PATH)


class MyModel(torch.nn.Module):
    """
    Simple dense layer with 4 inputs and 2 outputs.
    """

    def __init__(self):
        """
        Initialize the model
        """
        super(MyModel, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(4, 2))
        self.bias = torch.nn.Parameter(torch.randn(2))

    def forward(self, inp):
        return torch.sigmoid(torch.matmul(inp, self.weights) + self.bias)


model = MyModel()

# create random input with fixed seed
np.random.seed(0)
input_data = np.random.randn(1, 4).astype(np.float32)
predictions = model(torch.from_numpy(input_data)).detach().numpy()

# save predictions to results/py.json
with open(os.path.join(RESULT_PATH, 'py.json'), 'w', encoding='utf-8') as f:
    f.write(str(predictions.tolist()))

# save model
torch.save(model.state_dict(), FOLDER_PATH)

inputs = torch.tensor(np.random.randn(1, 4).astype(np.float32))

torch.onnx.export(model, inputs, f'{FOLDER_PATH}.onnx', export_params=True, opset_version=11, do_constant_folding=True,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

netron.start(f'{FOLDER_PATH}.onnx')
