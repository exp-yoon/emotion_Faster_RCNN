import torch
import torch.nn as nn
import numpy as np
from util import IoU


def proposal_target_layer(roi,gt_box):
    
    n_sample = 128
    iou_threshold = 0.5
    pos_ratio = 0.25
    neg_ratio = 0.75
    IoU_list = IoU(roi,gt_box,batch_size)
    
    roi_label = np.empty((batch_size,2000))
    roi_label.fill(-1)
    
    for b in range(batch_size):
        for i in range(2000):
                                    
            if IoU_list[b,i] >= iou_threshold:
                roi_label[b,i] = 1
                                                                    
            elif IoU_list[b,i] < iou_threshold:
                roi_label[b,i] = 0

    #mini_batch                                           
    for b in range(batch_size):
        pos_index = np.where(roi_label[b] == 1)[0]
        neg_index = np.where(roi_label[b] == 0)[0]

        if len(pos_index) > n_sample * pos_ratio :
            disable_idx = np.random.choice(pos_index,size = (len(pos_index)-(n_sample * pos_ratio)),replace = False)
            roi_label[b,disable_idx] = -1                                
        
        if len(neg_index) > n_sample * neg_ratio: 
            disable_idx = np.random.choice(neg_index, size = (len(neg_index)-(n_sample * neg_ratio)),replace = False)
            roi_label[b,disable_idx] = -1

        if len(pos_index) < n_sample * pos_ratio:
            add_idx = np.random.choice(neg_index, size = ((n_sample * pos_ratio) - len(pos_index)),replace = False)
            roi_label[b,add_idx] = 1


    return roi_label
