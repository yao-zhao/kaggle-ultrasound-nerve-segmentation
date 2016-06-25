# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 18:58:44 2016
create run length
@author: yz
"""
import numpy as np

def run_length(mask, seg_threshold):
    mask = mask.transpose().flatten()
    out = []    
    if mask[0]>seg_threshold:
        start_ind = 0
    else:
        start_ind = -1
    run_length = 0
    for index, pix in zip(range(len(mask)),mask):
        if pix>seg_threshold:
            if start_ind<0:
                start_ind = index
            else:
                run_length += 1
        else:
            if not np.isnan(start_ind):
                out.append(start_ind)
                out.append(run_length)
                start_ind = -1
                run_length = 0
    return out;
    
from itertools import chain;
def run_length_alex(label, seg_threshold):
    x = label.transpose().flatten();
    y = np.where(x>seg_threshold)[0];
    if len(y)<10:# consider as empty
        return [];
    z = np.where(np.diff(y)>1)[0]
    start = np.insert(y[z+1],0,y[0])
    end = np.append(y[z],y[-1])
    leng = end - start;
    res = [[s+1,l+1] for s,l in zip(list(start),list(leng))]
    res = list(chain.from_iterable(res));
    return res;