# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:16:21 2016
label the training images with BP or no BP 
and saves the result 
this serves as the first part of image segmentation
@author: yz
"""

import sys
caffe_root = '/home/yz/caffe-yao'
sys.path.insert(0, caffe_root + '/python')
import os
import cv2
import csv
import sort_human


root_path = '/home/yz/uns/'
data_path = root_path + 'data/'

train_path =  data_path + 'raw/train/'
train_file = data_path + 'train.txt'
train_BP_file = data_path + 'train_BP_label.txt'
train_seg_file = data_path + 'train_seg.txt'

val_file = data_path + 'val.txt'
val_BP_file = data_path + 'val_BP_label.txt'
val_seg_file = data_path + 'val_seg.txt'

test_path = data_path+'raw/test/'
test_BP_file = data_path + 'test_BP_label.txt'
test_seg_file = data_path + 'test_seg.txt'

sourcefile = train_file
targetBP = train_BP_file
targetSeg  = train_seg_file
for sourcefile, targetBP, targetSeg in \
    zip([train_file, val_file],
        [train_BP_file, val_BP_file],
        [train_seg_file, val_seg_file]):
    filenames = []
    masknames = []
    hasBP =[]
    # read all training images and check if the mask is empty
    with open(sourcefile,'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            img = cv2.imread(train_path+row[1], flags=0)
            filenames.append(row[0])
            masknames.append(row[1])
            hasBP.append(int(img.sum()>0))
    with open(targetBP,'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for fi, bp in zip(filenames,hasBP):
            writer.writerow([fi, str(bp)]);
    with open(targetSeg,'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for fi, bp, mask in zip(filenames,hasBP, masknames):
            if bp:
                writer.writerow([fi,mask])

# generate bp label and seg label for test path
# sort the files by their order in number
files = os.listdir(test_path)
sort_human.sort(files)
numfiles = len(files)
with open(test_BP_file,'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for index in range(len(files)):    
        if files[index].find('mask')<=0:
            writer.writerow([files[index], str(0)])      

with open(test_seg_file,'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for index in range(len(files)):    
        if files[index].find('mask')<=0:
            writer.writerow([files[index], 'empty_mask.tif'])      
