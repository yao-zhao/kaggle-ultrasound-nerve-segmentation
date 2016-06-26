# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:01:17 2016

@author: yz
"""
import sys
caffe_root = '/home/yz/caffe-yao'
sys.path.insert(0, caffe_root + '/python')
import os
import cv2
import numpy as np
import sort_human
import argparse
import csv
import re

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

patient_id = []
img_id = []
for filename in files:
    matched = re.match('(\d*)_(\d*)\.tif',filename)
    if matched:
        patient_id.append(int(matched.group(1)))
        img_id.append(int(matched.group(2)))
unique_patient_id = np.unique(patient_id)
id_train_val_bound = len(unique_patient_id)*(train_val_ratio-1)/train_val_ratio

print 'following patients in training set:'
for pid in unique_patient_id:
    if (pid) < id_train_val_bound:
        print(str(pid)),
        
print '\nfollowing patients in validation set:'
for id in unique_patient_id:
    if id >= id_train_val_bound:
        print str(id),
print ''
             
train_files = []
val_files = []
for i in range(numfiles/2):
    matched = re.match('(\d*)_(\d*)\.tif',files[2*i])
    if matched:
        pid = int(matched.group(1))
    else:
        print 'something is wrong here'
    if pid >= id_train_val_bound:
        val_files.append(files[2*i])
        val_files.append(files[2*i+1])
    else:
        train_files.append(files[2*i])
        train_files.append(files[2*i+1])

with open(train_file,'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for index in range(len(train_files)/2):    
        writer.writerow([train_files[2*index],train_files[2*index+1]])        

with open(val_file,'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for index in range(len(val_files)/2):    
        writer.writerow([val_files[2*index],val_files[2*index+1]])        

