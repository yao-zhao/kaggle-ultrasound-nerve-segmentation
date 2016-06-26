# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:58:59 2016

@author: yz
"""
import os
import cv2
import numpy as np
import sort_human
root_path = '/home/yz/uns/'
data_path = root_path + 'data/'
train_path =  data_path + 'raw/train/'

# sort the files by their order in number
files = os.listdir(train_path)
sort_human.sort(files)
numfiles = len(files)

images=[]
segs=[]
for i in range(numfiles/2):
    images.append(files[2*i])
    segs.append(files[2*i+1])
    
    
for i in range(numfiles/2):
    img = cv2.imread(train_path+images[i],flags=0)    
    seg = cv2.imread(train_path+segs[i],0)*255
    ret,thresh = cv2.threshold(seg,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    filename=images[i]
    cv2.namedWindow(filename)
    cv2.moveWindow(filename,10,50)
    cv2.imshow(filename,img)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)