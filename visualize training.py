# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:14:14 2016
visualize logs
@author: yz
"""
import matplotlib.pyplot as plt
#import parseLog
import sys
caffe_root = '/home/yz/caffe_yao'
sys.path.insert(0, caffe_root + '/python')
log = '/home/yz/uns/models/segnet/all64/log.txt'
log2 = '/home/yz/uns/models/segnet/batchnorm1D//log.txt'



result = parseLog(log)
train = result['train']
test = result['test0']
plt.plot(train['iter'],train['total_loss'],'b')
plt.plot(test['iter'],test['loss'],'g')
result = parseLog(log2)
train = result['train']
test = result['test0']
plt.plot(train['iter'],train['total_loss'],'r')
plt.axis([0,6000,0, 0.2])
plt.show()