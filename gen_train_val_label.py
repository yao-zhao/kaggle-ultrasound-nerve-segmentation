# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:01:17 2016

@author: yz
"""
import os
import cv2
import numpy as np
import sort_human
import argparse
import sys
import csv

root_path = '/home/yz/uns/'
data_path = root_path + 'data/'
train_path =  data_path + 'raw/train/'
train_file = data_path + 'train.txt'
val_file = data_path + 'val.txt'

# sort the files by their order in number
files = os.listdir(train_path)
sort_human.sort(files)
numfiles = len(files)

# parse input
parser = argparse.ArgumentParser(description='Process some image dimensions')
parser.add_argument('--TRAIN_VAL_RATIO', type=float, default=8)
parsed = parser.parse_args(sys.argv[1:])
train_val_ratio = parsed.TRAIN_VAL_RATIO

train_files = []
val_files = []
for i in range(numfiles/2):
    if i % train_val_ratio == 0:
        val_files.append(files[2*i])
        val_files.append(files[2*i+1])
    else:
        train_files.append(files[2*i])
        train_files.append(files[2*i+1])

with open(train_file,'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for index in range(len(train_files)):    
        writer.writerow([train_files[index]])        

with open(val_file,'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for index in range(len(val_files)):    
        writer.writerow([val_files[index]])        

