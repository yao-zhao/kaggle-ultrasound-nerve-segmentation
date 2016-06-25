# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:14:14 2016
visualize logs
@author: yz
"""
import matplotlib
import matplotlib.pyplot as plt
#import parseLog
import sys
import parseLog
caffe_root = '/home/yz/caffe-yao'
sys.path.insert(0, caffe_root + '/python')
#log = '/home/yz/uns/models/segnet/all64/log.txt'
#log2 = '/home/yz/uns/models/segnet/batchnorm1D//log.txt'
#log = '/home/yz/uns/models/segnet/all32 deep norelu/log2.txt'
#log2 = '/home/yz/uns/models/segnet/exp32 deep/log.txt'
log = '/home/yz/uns/models/segnet/exp32 deep bponly/log.txt'
log2 = '/home/yz/uns/models/segnet/net2/log.txt'
#log2 = '/home/yz/uns/models/segnet/all32 deep/log.txt'

#pylab.show()
result = parseLog.parseLog(log)
train = result['train']
test = result['test0']

plt.figure()
plt.show()
plt.plot(train['iter'],train['total_loss'],'b')
plt.plot(test['iter'],test['loss'],'g')
result = parseLog.parseLog(log2)
train = result['train']
test = result['test0']
plt.plot(train['iter'],train['total_loss'],'r')
plt.plot(test['iter'],test['loss'],'y')
plt.axis([0,4000,0, 0.1])
plt.legend([log, log2])
plt.savefig('/home/yz/uns/models/segnet/compare.png')
