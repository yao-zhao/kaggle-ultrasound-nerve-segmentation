# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:01:17 2016

@author: yz
"""
import os
import cv2
import sort_human

root_path = '/home/yz/uns/'
data_path = root_path + 'data/'
train_path =  data_path + 'raw/train/'
train_file = data_path + 'train.txt'
val_file = data_path + 'val.txt'

# sort the files by their order in number
files = os.listdir(train_path)
sort_human.sort(files)
numfiles = len(files)

image_files = []
mask_files = []
for i in range(numfiles/2):
    image_files.append(files[2*i])
    mask_files.append(files[2*i+1])

for filename in mask_files:
    img = cv2.imread(train_path+filename,flags=0)
    break
#    img =img/255
#    cv2.imwrite(train_path+filename,img)
