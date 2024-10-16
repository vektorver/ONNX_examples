"""
This script generates random images and saves them in a folder.
Format of data:
    X_test: [AMOUNT, HEIGHT, WIDTH, CHANNELS]
"""
import os
import shutil

import numpy as np

images = np.random.rand(5, 88, 120, 3) * 255

FOLDER_NAME = 'data'

# create folder if not exists
if os.path.exists(FOLDER_NAME):
    shutil.rmtree(FOLDER_NAME)

os.mkdir(FOLDER_NAME)

with open(os.path.join(FOLDER_NAME, 'X_test.json'), encoding='utf-8', mode='w') as f:
    f.write(str(images.tolist()))
