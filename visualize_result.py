# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:58:59 2016

@author: yz
"""
import sys
import cv2
import numpy as np
import os
root_path = '/home/yz/uns/'
data_path = root_path + 'data/'
train_path =  data_path + 'raw/train/'
caffe_path = '/home/yz/caffe-yao'
sys.path.insert(0, caffe_path + '/python')
import caffe
import argparse
os.chdir(root_path)
import sort_human

# parse input
parser = argparse.ArgumentParser(description='Process some image dimensions')
parser.add_argument('--MODEL_WEIGHT', type=str,default='data/models/segnet_iter_10000.caffemodel')
parser.add_argument('--MODEL_DEF', type=str,default="models/segnet/deploy.prototxt")

parsed = parser.parse_args(sys.argv[1:])
model_weights = root_path + parsed.MODEL_WEIGHT
model_def = root_path + parsed.MODEL_DEF


# sort the files by their order in number
files = os.listdir(train_path)
sort_human.sort(files)
numfiles = len(files)

# load net
net = None
net = caffe.Net(model_def,model_weights,caffe.TEST)
net.forward()
i=0
data = net.blobs['data'].data
label = net.blobs['label'].data
score = net.blobs['loss'].data

img = np.transpose(data[i,:,:,:],(1,2,0)).astype(np.uint8)[:,:,0]
img2 = np.transpose(label[i,:,:,:],(1,2,0)).astype(np.uint8)[:,:,0]
ret,thresh = cv2.threshold(img2,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)
filename='img'
cv2.namedWindow(filename)
cv2.moveWindow(filename,10,50)
cv2.imshow(filename,img)   
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range (1,5):
    cv2.waitKey(1)

#images=[]
#segs=[]
#for i in range(numfiles/2):
#    images.append(files[2*i])
#    segs.append(files[2*i+1])
#    
#    
#for i in range(20):
#    img = cv2.imread(train_path+images[i],flags=0)    
#    seg = cv2.imread(train_path+segs[i],0)
#    ret,thresh = cv2.threshold(seg,127,255,0)
#    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    cv2.drawContours(img, contours, -1, (0,255,0), 3)
#    filename=images[i]
#    cv2.namedWindow(filename)
#    cv2.moveWindow(filename,10,50)
#    cv2.imshow(filename,img)   
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    for i in range (1,5):
#        cv2.waitKey(1)