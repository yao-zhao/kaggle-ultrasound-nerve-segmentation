# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:14:14 2016
visualize logs
@author: yz
"""
import matplotlib
import matplotlib.pyplot as plt
import sys
caffe_root = '/home/yz/caffe-yao'
sys.path.insert(0, caffe_root + '/python')
import parseLog

log = '/home/yz/uns/models/bpnet/net11/log.txt'
log2 = '/home/yz/uns/models/bpnet/net12/log.txt'


plt.figure()
plt.show()

result = parseLog.parseLog(log)
train = result['train']
test = result['test0']
plt.plot(train['iter'],train['total_loss'],'b')
plt.plot(test['iter'],test['loss'],'g')

result = parseLog.parseLog(log2)
train = result['train']
test = result['test0']
plt.plot(train['iter'],train['total_loss'],'r')
plt.plot(test['iter'],test['loss'],'y')

plt.axis([0,4000,0.1, 0.8])
plt.legend([log, log2])
plt.savefig('/home/yz/uns/models/bpnet/compare.png')
