# Get file keys and save in a *.txt for use

import sys
from PIL import Image
import numpy as np
import os, os.path
from os.path import isfile, join
import matplotlib.pyplot as plt
import math

# file_dir = "/Users/wxwang/Desktop/DL/Kaggle_p1/im1.png"
# im = Image.open(file_dir)
# im.show()
# pixels = im.load()
# a, b = im.size
# r = pixels[1, 1]
# print(pixels[1, 1])
# print(r[1])
# print(len(os.listdir(file_dir2)))

# print(a, b)

data_dir = "stage1_train"
count = 1
with open("image_train.txt", "w") as image_train, \
        open("class_train.txt", "w") as class_train, \
        open("segmentation_train.txt", "w") as segmentation_train:
    for dir in sorted(os.listdir(data_dir))[0: 2]:
        if dir == '.DS_Store':
            continue
        image_dirs = os.path.join(data_dir, dir, "images")
        image_location = os.listdir(image_dirs)
        image = Image.open(os.path.join(image_dirs, image_location[0]))
        image_train.write(os.path.join(image_dirs, image_location[0]) + "\n")
        class_train.write("1\n")
        segmentation_dirs = os.listdir(os.path.join(data_dir, dir, "masks"))
        for i, segmentation_dir in enumerate(segmentation_dirs):
            segmentation_train.write(os.path.join(data_dir, dir, "masks", segmentation_dir))
            if i < len(segmentation_dirs) - 1:
                segmentation_train.write(',')
        segmentation_train.write("\n")
