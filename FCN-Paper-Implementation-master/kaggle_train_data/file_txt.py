# Get file keys and save in a *.txt for use

import sys
from PIL import Image
import numpy as np
import os, os.path
from os.path import isfile,join
import matplotlib.pyplot as plt
import math

file_dir = "/Users/wxwang/Desktop/DL/Kaggle_p1/im1.png"
file_dir2 = "/Users/wxwang/Desktop/DL/Kaggle_p1/stage1_train"
im = Image.open(file_dir)
#im.show()
pixels = im.load()
a,b = im.size
r = pixels[1,1]
print(pixels[1,1])
print(r[1])
print(len(os.listdir(file_dir2)))



print(a,b)
count = 1
filekey = open("train_kaggle1.txt","w")
filekey2 = open("train_kaggle1_class.txt","w")
filekey3 = open("train_kaggle1_segm.txt","w")
for i in os.listdir(file_dir2)[0:1]:
    file_dir3 = os.path.join(file_dir2,i,"Images")
    file_dir4 = os.listdir(file_dir3)
    im = Image.open(os.path.join(file_dir3,file_dir4[0]))
    pixels = im.load()
    print(os.path.join(file_dir3,file_dir4[0]))
    filekey.write(os.path.join(file_dir3,file_dir4[0])+"\n")
    filekey2.write("1\n")
    file_dir5 = os.listdir(os.path.join(file_dir2,i,"masks"))
    for ii in file_dir5:
        filekey3.write(os.path.join(file_dir2,i,"masks",ii)+",")
    filekey3.write("\n")
filekey.close()



