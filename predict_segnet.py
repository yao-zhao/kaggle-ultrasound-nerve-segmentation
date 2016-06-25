# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 19:39:44 2016
classify results using bp first and sec next
@author: yz
"""

# inference BP or no BP



caffe_root = '/home/yz/caffe-yao/'  
root='/home/yz/uns/'
solver_state_path = root+'/data/models/'
testpath = root + "data/raw/test/"
testlmdb = root + "data/test_lmdb/"
testfile = root + 'data/test_seg.txt'

import pylab
import numpy as np
import cv2
import os
import sys
sys.path.insert(0, caffe_root + '/python')
import caffe
os.chdir(root)
import csv
import argparse
import time


# parse input
parser = argparse.ArgumentParser(description='Process testing data')
parser.add_argument('--MODEL_WEIGHT', type=str,default='data/models/segnet_iter_4000.caffemodel')
parser.add_argument('--MODEL_DEF', type=str,default="models/segnet/deploy.prototxt")
parser.add_argument('--RESULT_NAME', type=str,default='0')
# use parse
parsed = parser.parse_args(sys.argv[1:])
model_weights = root + parsed.MODEL_WEIGHT
model_def = root + parsed.MODEL_DEF
resultfolder = root+'data/result_mask/result'+parsed.RESULT_NAME+'/'
if not os.path.exists(resultfolder):
	os.mkdir(resultfolder)
print parsed

# read test file names
testfilenames = []
with open(testfile, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter = ' ')
    for row in reader:
        testfilenames.append(row[0])
num_test=len(testfilenames)
# prepare net
net = None
net = caffe.Net(model_def,model_weights,caffe.TEST)
batch_size,numchannel,crop_h,crop_w = net.blobs['data'].data.shape
caffe.set_mode_gpu()
# calculate batches
fi = 0
num_batches = np.floor(num_test/batch_size).astype(np.int)+1
prob = np.zeros((batch_size*num_batches,crop_h,crop_w),np.float)
cvimg = np.zeros((crop_h,crop_w,3),np.uint8)
for ibatch in range(num_batches):
    print [pylab.double(ibatch)/num_batches, batch_size]
    output = net.forward()
    prob[ibatch*batch_size:(ibatch+1)*batch_size,:,:] = output['prob'][:,1,:,:]
    for i in range(batch_size):
        if fi < num_test:
            img = (output['prob'][i,1,:,:] * 255).astype(np.uint8)
            cvimg[:,:,0] = img
            cv2.imwrite('mask_'+testfilenames[fi], cvimg)
            fi += 1
prob=prob[0:num_test,:,:]
np.save(resultfolder+'segprob.npy',prob)




