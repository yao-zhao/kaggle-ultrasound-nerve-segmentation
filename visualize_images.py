# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:58:59 2016

@author: yz
"""
import os
import cv2
import numpy as np
import sort_human
root_path = '/home/yz/uns/'
data_path = root_path + 'data/'
train_path =  data_path + 'raw/train/'

# sort the files by their order in number
files = os.listdir(train_path)
sort_human.sort(files)