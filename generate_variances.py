#!/usr/bin/env python3

#General Packages
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Packages to make life easier
from tqdm import tqdm, trange
import yaml, glob, pickle, sys
from os import getcwd


def interpolate_data(last_pt, pt, res, origin, num_pts):
    last_x = last_pt.position.x - origin[0]
    last_y = last_pt.position.y - origin[1]
    last_x_coord = int(last_x/res)
    last_y_coord = h - int(last_y/res)

    x = pt.position.x - origin[0]
    y = pt.position.y - origin[1]
    x_coord = int(x/res)
    y_coord = h - int(y/res)

    x = np.linspace(last_x_coord, x_coord, num_pts)
    if x_coord-last_x_coord != 0:
        #If this moves laterally, we can make a slope and generate y data
        m = -(y_coord - last_y_coord)/(x_coord - last_x_coord)
    else:
        if y_coord-last_y_coord == 0:
            #If there is no delta x and no delta y, this is just all one point
            return [(x_coord, y_coord)]*num_pts
        else:
            #Otherwise this is a vertical line, so we need to interpolate
            #   between the y points
            y = np.linspace(last_y_coord, y_coord, num_pts)
            return list(zip(x.astype(int), y.astype(int)))
    y = -np.subtract(m*np.subtract(x, last_x_coord), last_y_coord).astype(int)

    return list(zip(x.astype(int), y))

if __name__ == "__main__":
    path = getcwd() + '/../ebola_final/'
    data_path = path + 'etu_1_condensed/'

    num_pts = 100

    map_params = yaml.load(open(path + 'map.yaml', 'rb'))
    origin = map_params['origin']
    res = map_params['resolution'] #meters per pixel
    I = Image.open(path + 'map.pgm')
    imarray = np.asarray(I).astype('int')
    w, h = I.size

    """
    im = [[[] for k in range(w)] for i in range(h)]

    files = glob.glob(data_path + '*.p')

    tqdm.write("Generating variance data")
    #Generate heatmap
    for file in tqdm(files, position = 0):
        dataset = pickle.load(open(file, 'rb'))
        pt_dict = {}
        for trial in dataset:
            if trial[0] == 4:
                continue

            interpolated_data = []

            for i, point in enumerate(trial[2]):
                x = point.position.x - origin[0]
                y = point.position.y - origin[1]
                x_coord = int(x/res)
                y_coord = h - int(y/res)

                if i == 0:
                    interpolated_data.append((x_coord, y_coord))
                    continue
                interpolated_data.extend(interpolate_data(trial[2][i-1], point, res, origin, num_pts))

            for point in interpolated_data:
                if point in pt_dict.keys():
                    pt_dict[point] += 1
                else:
                    pt_dict[point] = 1

        for pt, val in pt_dict.items():
            im[pt[1]][pt[0]].append(val)

    np.save("time_in_cell", im)
    # """

    im = np.load("time_in_cell.npy")

    tqdm.write("Getting means, medians, and standard deviations")
    mean_data = np.zeros((w,h), dtype=int)
    median_data = np.zeros((w,h), dtype=int)
    std_data = np.zeros((w,h), dtype=int)
    for i in trange(h):
        for j in range(w):
            if len(im[i][j]) != 0:
                mean_data[i][j] = np.mean(im[i][j])
                median_data[i][j] = np.median(im[i][j])
                std_data[i][j] = np.std(im[i][j])

    np.save("mean_heatmap", mean_data)
    np.save("median_heatmap", median_data)
    np.save("std_heatmap", std_data)

    tqdm.write("Generating log-scaled images")
    #Create a version of the heatmap on a log scale
    for i in trange(h):
        for j in range(w):
            if mean_data[i][j] > 0:
                mean_data[i][j] = np.log(mean_data[i][j])
                median_data[i][j] = np.log(median_data[i][j])
            if std_data[i][j] > 0:
                std_data[i][j] = np.log(std_data[i][j])

    plt.imsave("mean_heatmap.png", mean_data, cmap=plt.cm.plasma)
    plt.imsave("median_heatmap.png", median_data, cmap=plt.cm.plasma)
    plt.imsave("std_heatmap.png", std_data, cmap=plt.cm.plasma)

    # plt.imsave("heatmap_log", im_log, cmap=plt.cm.Reds)
    # plt.imshow(im_log, cmap=plt.cm.Reds, interpolation='nearest')
    # plt.show()
