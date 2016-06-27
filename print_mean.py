# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:23:28 2016

@author: yz
"""

root = '/home/yz/uns/'
caffe_root = '/home/yz/caffe-yao/'
import sys
sys.path.insert(0, caffe_root+'python')
import caffe
import numpy as np
import subprocess
tmp = root+'proto.mean'
lmdb = root+'data/train_lmdb'
subprocess.call('/home/yz/caffe-yao/build/tools/compute_image_mean'+
    ' '+lmdb+' '+tmp,shell=True)

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(tmp,'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
subprocess.call('rm '+tmp,shell=True)
print 'The average of pixel value of each channels are'+str(arr[0].mean(1).mean(1))

