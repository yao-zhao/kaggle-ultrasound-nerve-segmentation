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

data = net.blobs['data'].data
label = net.blobs['label'].data
score = net.blobs['loss'].data

batch_size, num_channels, height, width = data.shape
for i in range(batch_size):
    img_merge = np.zeros((height,width,3),np.uint8)
    
    img = np.transpose(data[i,:,:,:],(1,2,0)).astype(np.uint8)[:,:,0]+100
    img_merge[:,:,0]=img
    img_merge[:,:,1]=img
    img_merge[:,:,2]=img
    
    img2 = np.transpose(label[i,:,:,:],(1,2,0)).astype(np.uint8)[:,:,0]*255
    ret,thresh = cv2.threshold(img2,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_merge, contours, -1, (0,255,0), 3)
    
    img3 = (score[i,1,:,:]*255).astype(np.uint8)
    ret,thresh = cv2.threshold(img3,255*0.12,255,0)
    im3, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_merge, contours, -1, (0,0,255), 3)
#    img_merge[:,:,2]=img3
    
    filename='img'
    cv2.namedWindow(filename)
    cv2.moveWindow(filename,10,50)
    cv2.imshow(filename,img_merge)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)
    
    
#img3 = (np.transpose(score[i,0,:,:]*255)).astype(np.uint8)
#cv2.namedWindow(filename)
#cv2.moveWindow(filename,10,50)
#cv2.imshow(filename,img3)   
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#for i in range (1,5):
#    cv2.waitKey(1)
    
    
