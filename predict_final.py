# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:22:23 2016
use the predicted bp net and predicted seg net to create the final output
@author: yz
"""


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
import run_length

# get timer
tm = time.gmtime()
timestamp = str(tm.tm_year)+'_'+str(tm.tm_mon)+'_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+str(tm.tm_min)+'_'+str(tm.tm_sec)
# parse input
parser = argparse.ArgumentParser(description='Process combine BP detection and mask segmentation')
parser.add_argument('--BP_PREDICTION', type=str,default='results/result_1.csv')
parser.add_argument('--MASK_FOLDER', type=str,default='data/result_mask/result_1/')
parser.add_argument('--RESULT_NAME', type=str,default='0')
parser.add_argument('--BP_THRESHOLD', type=float, default=0.5)
#parser.add_argument('--USE_BALANCE', type=int, default=1)
parser.add_argument('--SEG_THRESHOLD', type=float, default=0.5)
# use parse
parsed = parser.parse_args(sys.argv[1:])
bp_file = root + parsed.BP_PREDICTION
seg_folder = root + parsed.MASK_FOLDER
resultfile = root+'results/submission_'+parsed.RESULT_NAME+'.txt'
bp_threshold = parsed.BP_THRESHOLD
seg_threshold = parsed.SEG_THRESHOLD
#if parser.USE_BALANCE==0:
#    use_balance = False
#else:
#    use_balance = True
print parsed

# load bp probability
filenames = []
bp_probs = []
with open(bp_file,'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        filenames.append(row[0])
        bp_probs.append(float(row[2]))
        
# find counts for bp and no bp
count_bp = 0
count_nobp = 0
for pb in bp_probs:
    if pb > bp_threshold:
        count_bp += 1
    else:
        count_nobp += 1
print 'find '+str(count_bp)+ ' images with BP, and '+str(count_nobp)+' images without BP'

# infer mask size and raw size
raw_height, raw_width, num_channel = cv2.imread(testpath+os.listdir(testpath)[0]).shape
mask_height, mask_width, num_channel = cv2.imread(seg_folder+os.listdir(seg_folder)[0]).shape
pad_height = 224
pad_width = 310
padimg = np.zeros((pad_height,pad_width,3),np.uint8)
h_off = (pad_height - mask_height)/2
w_off = (pad_width - mask_width)/2
full_mask = np.zeros((raw_height,raw_width,3),np.uint8)
# go through each picture
count = 0 
with open(resultfile,'wb') as f:
    f.write('img,pixels\n')
    for bp, filename in zip(bp_probs, filenames):
        print filename
        f.write(filename[0:-4]+',')
        # if detected bp    
        if bp > bp_threshold:
            rawimg = cv2.imread(testpath+filename)
            maskimg = cv2.imread(seg_folder+'mask_'+filename)
            padimg[h_off:h_off+mask_height, w_off:w_off+mask_width,0]=maskimg[:,:,0]
            full_mask = cv2.resize(padimg,(raw_width,raw_height))
            # write file name            
            for j in run_length.run_length_alex(full_mask[:,:,0], seg_threshold):
                f.write(str(j)+' ')
        f.write('\n')
            
    #    else:
    #        rawimg = cv2.imread(testpath+filename)
    #        full_mask = np.zeros((raw_height,raw_width,3),np.uint8)
    #    count += 1
    #    imgshow = rawimg
    #    ret,thresh = cv2.threshold(full_mask[:,:,0],seg_threshold*255,255,0)
    #    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #    cv2.drawContours(imgshow, contours, -1, (0,255,0), 2)
    #    filename='img'
    #    cv2.namedWindow(filename)
    #    cv2.moveWindow(filename,10,50)
    #    cv2.imshow(filename,imgshow)   
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
    #    for i in range (1,5):
    #        cv2.waitKey(1)
    #    if count > 20:
    #        break
        

