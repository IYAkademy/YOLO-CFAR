# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 18:30:27 2022

@author: Paul
@file: resize_images.py
@dependencies:
    env pt3.7
    python 3.7.13
    torch >= 1.7.1
    torchvision >= 0.8.2
    Pillow >= 8.1.0

Resize image to a certain size
"""
# import the required libraries
import torchvision.transforms as T # for resizing the images
from PIL import Image              # for loading and saving the images

import json
import time
import os
from os import listdir

# set the dataset path
DATASET = 'D:/Datasets/CARRADA/'
DATASET2 = 'D:/Datasets/CARRADA2/'

# directory names, number of directorie: 30
dir_names = ['2019-09-16-12-52-12', '2019-09-16-12-55-51', '2019-09-16-12-58-42', '2019-09-16-13-03-38', '2019-09-16-13-06-41', 
             '2019-09-16-13-11-12', '2019-09-16-13-13-01', '2019-09-16-13-14-29', '2019-09-16-13-18-33', '2019-09-16-13-20-20', 
             '2019-09-16-13-23-22', '2019-09-16-13-25-35', '2020-02-28-12-12-16', '2020-02-28-12-13-54', '2020-02-28-12-16-05', 
             '2020-02-28-12-17-57', '2020-02-28-12-20-22', '2020-02-28-12-22-05', '2020-02-28-12-23-30', '2020-02-28-13-05-44', 
             '2020-02-28-13-06-53', '2020-02-28-13-07-38', '2020-02-28-13-08-51', '2020-02-28-13-09-58', '2020-02-28-13-10-51', 
             '2020-02-28-13-11-45', '2020-02-28-13-12-42', '2020-02-28-13-13-43', '2020-02-28-13-14-35', '2020-02-28-13-15-36']

# number of images / labels in each directory, total number of labels: 7193
num_of_images = [286, 273, 304, 327, 218, 219, 150, 208, 152, 174, 
                 174, 235, 442, 493, 656, 523, 350, 340, 304, 108, 
                 129, 137, 171, 143, 104, 81, 149, 124, 121, 98]

# test the basic functionality of resizing an image to certain size
def testing(i=1, file_type='jpg'):

    # read the input image
    # img = Image.open(f'D:/Datasets/RD_maps/scaled_colors/{i}_sc.png')
    img = Image.open(f'D:/Datasets/RD_maps/scaled_colors/{i}_sc.{file_type}')

    # compute the size (width, height) of image
    before = img.size
    print(f"original image size: {before}")

    # define the transform function to resize the image with given size, say 416-by-416
    transform = T.Resize(size=(416,416))

    # apply the transform on the input image
    img = transform(img)

    # check the size (width, height) of image
    after = img.size
    print(f"resized image size: {after}")

    # overwrite the original image with the resized one
    # img = img.save(f'D:/Datasets/RD_maps/scaled_colors/{i}_sc.png')
    img.show()


def main(max_iter=1, file_type='jpg'):
    # 1600
    for i in range(1, max_iter + 1):
        # read the input image
        img = Image.open(f'D:/Datasets/RD_maps/scaled_colors/{i}_sc.{file_type}')

        # define the transform function to resize the image with given size, say 416-by-416
        transform = T.Resize(size=(416,416))

        # apply the transform on the input image
        img = transform(img)

        # overwrite the original image with the resized one
        img = img.save(f'D:/Datasets/RD_maps/scaled_colors/{i}_sc.{file_type}')
        print(f"{i}")

if __name__ == '__main__':
    # testing(1, 'jpg')
    # main(1600, 'jpg')

    for dir_name in dir_names[23:24]: # : # 
        path = DATASET2 + f'RA/{dir_name}/images/'
        print(f'current path: {path}')
        # print(type(path)) # <class 'str'>
        for images in os.listdir(path):
            print(images) # e.g. 000035.png, 000177.png
            # print(type(images)) # <class 'str'>
            
            # read the input image
            img = Image.open(path + images)
            # define the transform function to resize the image with given size, say 416-by-416
            transform = T.Resize(size=(416,416))
            # apply the transform on the input image
            img = transform(img)
            # overwrite the original image with the resized one
            img = img.save(path + images)

