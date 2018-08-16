#!/usr/bin/env python3

#Get rid of those annoying numpy/tensorflow warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import argparse, glob, csv
from os import getcwd
from skimage.measure import block_reduce

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

def are_neighbors(output_data, i, j):
    neighbor_list = [(i-1, j+1), (i, j+1), (i+1,j+1),
                    (i-1, j),              (i+1, j),
                    (i-1, j-1), (i, j-1), (i+1, j-1)]

    for point in neighbor_list:
        if output_data[point[0]][point[1]] != 0:
            return True
    return False

if __name__ == "__main__":
    path = getcwd() + '/'

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store',default=path+'in_sage/',
                        dest='inpath', help="set the input data folder")
    parser.add_argument('-o', action='store', default=path+'out_sage/',
                        dest='outpath', help="set the output data folder")

    results = parser.parse_args()
    inpath = results.inpath
    outpath = results.outpath

    #Size of one side of the scanning window
    kernel_size = 128

    #get ordered lists of files
    infiles = sorted(glob.glob(inpath + '*'))
    outfiles = sorted(glob.glob(outpath + '*'))
    print(infiles)
    assert len(infiles) == len(outfiles), "Mismatched number of files"

    for i in trange(len(infiles), position=0, smoothing=.9):
        I = Image.open(infiles[i])
        w, h = I.size
        p = np.asarray(I).astype('int')

        output_data = np.load(outfiles[i])

        input_data = []
        output_list = []

        for i in trange(h, position=1, smoothing=.9):
            for j in range(w):
                if p[i][j] >= 254 and (output_data[i][j] != 0 or are_neighbors(output_data, i, j)):
                    input_data.append(get_input_data(p, i, j, kernel_size, h, w))
                    output_list.append(output_data[i][j])

    np.save("indata",np.array(input_data))
    np.save("outdata", np.array(output_list))
