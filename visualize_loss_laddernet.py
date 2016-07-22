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

log = '/home/yz/uns/models/laddernet/net1/log.txt'
#log2 = '/home/yz/uns/models/bpnet/net16/log.txt'


plt.figure()
plt.show()

result = parseLog.parseLog(log)
train = result['train']
test = result['test0']
plt.plot(train['iter'],train['total_loss'],'b')
plt.plot(train['iter'][:-1],train['loss_supervised'],'g')
plt.plot(test['iter'],test['prob'],'g')
#
#result = parseLog.parseLog(log2)
#train = result['train']
#test = result['test0']
#plt.plot(train['iter'],train['total_loss'],'r')
#plt.plot(test['iter'],test['loss'],'y')

plt.axis([0,4000,0, 5])
#plt.legend([log, log2])
plt.savefig('/home/yz/uns/models/laddernet/compare_loss.png')

#
#plt.figure()
#plt.show()
#
#result = parseLog.parseLog(log)
#train = result['train']
#test = result['test0']
#plt.plot(train['iter'][:-1],train['accuracy'],'b')
#plt.plot(test['iter'],test['accuracy'],'g')
#
#result = parseLog.parseLog(log2)
#train = result['train']
#test = result['test0']
#plt.plot(train['iter'][:-1],train['accuracy'],'r')
#plt.plot(test['iter'],test['accuracy'],'y')

#plt.axis([0,4000,0.1, 1])
#plt.legend([log, log2], loc = 'best')
#plt.savefig('/home/yz/uns/models/bpnet/compare_accuracy.png')
