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
testfile = root + 'data/test_BP_label.txt'

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

# get timer
tm = time.gmtime()
timestamp = str(tm.tm_year)+'_'+str(tm.tm_mon)+'_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+str(tm.tm_min)+'_'+str(tm.tm_sec)
# parse input
parser = argparse.ArgumentParser(description='Process testing data')
parser.add_argument('--MODEL_WEIGHT', type=str,default='data/models/bpnet_iter_1500.caffemodel')
parser.add_argument('--MODEL_DEF', type=str,default="models/bpnet/deploy.prototxt")
parser.add_argument('--RESULT_NAME', type=str,default='0')
# use parse
parsed = parser.parse_args(sys.argv[1:])
model_weights = root + parsed.MODEL_WEIGHT
model_def = root + parsed.MODEL_DEF
resultfile = root+'results/result_'+parsed.RESULT_NAME+'.csv'
print parsed

# read test file
testfilenames = []
with open(testfile, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter = ' ')
    for row in reader:
        testfilenames.append(row[0])
num_test=len(testfilenames)

net = None
net = caffe.Net(model_def,model_weights,caffe.TEST)
batch_size,numchannel,crop_h,crop_w = net.blobs['data'].data.shape
caffe.set_mode_gpu()

num_batches = np.floor(num_test/batch_size).astype(np.int)+1
prob = np.zeros((num_batches*batch_size,2))

for ibatch in range(num_batches):
    print [pylab.double(ibatch)/num_batches, batch_size]
    output = net.forward()
    prob[ibatch*batch_size:(ibatch+1)*batch_size,:] = output['prob']
prob=prob[0:num_test,:]


with open(resultfile,'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for index in range(num_test):    
        row = [testfilenames[index]]
        for j in range(2):
            row.append(str(prob[index,j]))
        writer.writerow(row)

# stats
bias = .3
count0 = 0
count1 = 0
for i in range(len(testfilenames)):
    if prob[i,0]>prob[i,1]+bias:
        count0 += 1
    else:
        count1 += 1
print 'finished'
sys.exit()

# read test file
trainfilenames = []
with open(testfile, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter = ' ')
    for row in reader:
        trainfilenames.append(row[0])
num_train=len(trainfilenames)

net = None
net = caffe.Net(root+"models/bpnet/deploy2.prototxt",model_weights,caffe.TEST)
batch_size,numchannel,crop_h,crop_w = net.blobs['data'].data.shape
caffe.set_mode_gpu()

num_batches = np.floor(num_train/batch_size).astype(np.int)+1
prob = np.zeros((num_batches*batch_size,2))

for ibatch in range(num_batches):
    print [pylab.double(ibatch)/num_batches, batch_size]
    output = net.forward()
    prob[ibatch*batch_size:(ibatch+1)*batch_size] = output['prob']
prob=prob[0:num_train,:]
# stats
train_count0 = 0
train_count1 = 0
for i in range(len(trainfilenames)):
    if prob[i,0]>prob[i,1]:
        train_count0 += 1
    else:
        train_count1 += 1




