#!/usr/bin/env python3

#Get rid of those annoying numpy/tensorflow warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import keras, argparse
from PIL import Image
from keras.models import model_from_json
import numpy as np
from os import getcwd
from tqdm import trange, tqdm
from skimage.measure import block_reduce
import matplotlib.pylab as plt

def log_im(im):
    new = np.zeros(im.shape, dtype=float)
    for i in range(w):
        for j in range(h):
            if im[i][j] > 0:
                new[i][j] = np.log(im[i][j])
    return new

def get_input_data(p, i, j, kernel_size, h, w):
    min_i = int(i-(kernel_size)/2)
    max_i = int(i+(kernel_size)/2)
    min_j = int(j-(kernel_size)/2)
    max_j = int(j+(kernel_size)/2)

    data = []
    for k in range(min_i, max_i):
        row = [1 if k<0 or l<0 or k>=h or l>=w or p[k][l] <254 else 0 for l in range(min_j, max_j)]
        data.append(row)

    data = block_reduce(np.array(data), (4,4), np.max)
    return np.array(data)

# load json and create model
json_file = open('convnet1_std.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("convnet1_std.h5")

loaded_model.summary()

parser = argparse.ArgumentParser()
parser.add_argument('-i', action='store',default='map.pgm',
                    dest='filename', help="set the input data folder")

filename = parser.parse_args().filename

I = Image.open(filename)
p = np.asarray(I).astype('int')
w, h = I.size
im = np.zeros((w, h), dtype=int)

kernel_size = 128

for i in trange(h, position=0, smoothing=.9):
    for j in range(w):
        if p[i][j] >= 254:
            in_data = get_input_data(p, i, j, kernel_size, h, w)
            img_x, img_y = in_data.shape
            in_data = in_data.reshape(1, img_x, img_y, 1)
            val = loaded_model.predict(in_data, verbose=0)

            if val[0][0] > 0:
                im[i][j] = val[0][0]


np.save("std_prediction_mit_2011-01-18-06-37-58", im)
plt.imsave("model_prediction_std_mit_2011-01-18-06-37-58.png", log_im(im), cmap=plt.cm.plasma)
plt.imshow(aoi, cmap=plt.cm.plasma, interpolation='nearest')
plt.show()
