#!/usr/bin/env python3

import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import argparse, glob, csv
from os import getcwd

def get_input_data(p, i, j, kernel_size, h, w):
    min_i = int(i-(kernel_size-1)/2)
    max_i = int(i+(kernel_size-1)/2)
    min_j = int(i-(kernel_size-1)/2)
    max_j = int(i+(kernel_size-1)/2)

    data = []
    for k in range(min_i, max_i):
        row = [1 if k<0 or l<0 or k>=h or l>=w or p[k][l] <254 else 0 for l in range(min_j, max_j)]
        data.append(row)
        # for l in range(min_j, max_j):
        #     # if k != i and l != j:
        #     if k<0 or l<0 or k>=h or l>=w:
        #         data[k-min_i].append(1)
        #     elif p[k][l] >= 254:
        #         data[k-min_i].append(0)
        #     else:
        #         data[k-min_i].append(1)

    return np.array(data)

if __name__ == "__main__":
    path = getcwd() + '/'

    # open('indata.csv', 'w')
    # indata = csv.writer(open('indata.csv', 'a'))
    #
    # open('outdata.csv', 'w')
    # outdata = csv.writer(open('outdata.csv', 'a'))

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store',default=path+'in_sage/',
                        dest='inpath', help="set the input data folder")

    results = parser.parse_args()
    inpath = results.inpath

    #Size of one side of the scanning window
    kernel_size = 61

    #get ordered lists of files
    infiles = sorted(glob.glob(inpath + '*'))

    for i in trange(len(infiles), position=0):
        I = Image.open(infiles[i])
        w, h = I.size
        p = np.asarray(I).astype('int')

        input_data_len = kernel_size*kernel_size

        input_data = []
        input_points = []

        for i in trange(h, position=1):
            for j in range(w):
                if p[i][j] >= 254:
                    input_data.append(get_input_data(p, i, j, kernel_size, h, w))
                    # input_points.append((i, j))


    np.save("indata_for_prediction",np.array(input_data))
    # np.save("input_prediction_points", np.array(input_points))
