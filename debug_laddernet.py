# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:50:38 2016
debug ladder net
@author: yz
"""

caffe_root = '/home/yz/caffe-yao/'  
root='/home/yz/uns/'
solver_state_path = root + '/data/models/'
model_def = root + '/models/laddernet/train.prototxt'

import pylab
import numpy as np
import cv2
import os
import sys
sys.path.insert(0, caffe_root + '/python')
import caffe
os.chdir(root)

net = None
net = caffe.Net(model_def,caffe.TRAIN)
caffe.set_mode_gpu()
#caffe.set_device(1)

net.forward()



z1 = net.blobs['z1'].data[0,0,:,:]
z1_n = net.blobs['z1_n'].data[0,0,:,:]
z1_r = net.blobs['z1_r'].data[0,0,:,:]
bn1 = net.blobs['bn1_param'].data

z2 = net.blobs['z2'].data[0,0,:,:]
z2_n = net.blobs['z2_n'].data[0,0,:,:]
z2_r = net.blobs['z2_r'].data[0,0,:,:]
bn2 = net.blobs['bn2_param'].data

for i in range(2):
    data = net.blobs['data'].data[i,0,:,:]
    cv2.namedWindow('input')
    cv2.moveWindow('input',10,50)
#    img=np.zeros((3),np.uint8)
#    img[:,:,0]=data
    cv2.imshow('input',(data*255+100).astype(np.uint8))   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)