#!/usr/bin/env python3

#Get rid of those annoying numpy/tensorflow warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras

def unet (input_width, input_height, nChannels):
    model=Sequential()
    # inputs = Input((nChannels, input_height, input_width))
    model.add(Conv2D(input_width, kernel_size=(10, 10), strides=(3, 3),
                     activation='relu',
                     input_shape=(input_width, input_height, nChannels)))
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    #
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    Concatenate(axis=-1)
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # Concatenate(axis=-1)
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))

    model.summary()
    return model
