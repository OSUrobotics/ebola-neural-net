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
    last_y_coord = im_h - int(last_y/res)

    x = pt.position.x - origin[0]
    y = pt.position.y - origin[1]
    x_coord = int(x/res)
    y_coord = im_h - int(y/res)

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

    step = 1
    num_pts = 100

    map_params = yaml.load(open(path + 'map.yaml', 'rb'))
    origin = map_params['origin']
    res = map_params['resolution'] #meters per pixel
    I = Image.open(path + 'map.pgm')
    imarray = np.asarray(I).astype('int')
    w, h = I.size
    im_w = int(w/step)
    im_h = int(h/step)
    im = np.zeros((im_w, im_h), dtype=int)


    files = glob.glob(data_path + '*.p')

    tqdm.write("Generating variance data")
    #Generate heatmap
    for file in tqdm(files, position = 0):
        dataset = pickle.load(open(file, 'rb'))

        for trial in dataset:
            if trial[0] == 4:
                continue

            interpolated_data = []

            for i, point in enumerate(trial[2]):
                x = point.position.x - origin[0]
                y = point.position.y - origin[1]
                x_coord = int(x/res)
                y_coord = im_h - int(y/res)

                if i == 0:
                    interpolated_data.append((x_coord, y_coord))
                    continue
                interpolated_data.extend(interpolate_data(trial[2][i-1], point, res, origin, num_pts))

            for point in interpolated_data:
                im[point[1]][point[0]]+=1

    #Create a version of the heatmap on a log scale
    # im_log = np.zeros((im_w, im_h), dtype=int)
    # for i in trange(im_h):
    #     for j in range(im_w):
    #         if im[i][j] != 0:
    #             im_log[i][j] = np.log(im[i][j])

    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    from cv2 import imwrite
    imwrite("variance_heatmap.pgm", im)
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

    np.save("variance_heatmap", im)
    # plt.imsave("heatmap_log", im_log, cmap=plt.cm.Reds)
    # plt.imshow(im_log, cmap=plt.cm.Reds, interpolation='nearest')
    # plt.show()
