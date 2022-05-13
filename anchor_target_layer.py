import torch
import numpy as np
from anchor_generator import anchor_generator
from util import IoU
#inside anchor box select, calc IoU valid anchor & gt bbox
#sampling anchor box positive(>=0.7) , negative(<= 0.3), other(-1)

def anchor_target(anchor, bbox, area ,batch_size):
    anchor_box = anchor
    valid_anchor_box = []
    #print(valid_anchor_box)
    #valid anchor box select, img size->input
    for b in range(batch_size):
        index_inside = np.where(
                (anchor_box[b,:,0] >= 0) &
                (anchor_box[b,:,1] >= 0) &
                (anchor_box[b,:,2] <= 800) &
                (anchor_box[b,:,3] <= 800))[0]
        #print(anchor_box[b,index_inside])
        valid_anchor_box.append(anchor_box[b,index_inside]) #(index_inside num,4)   
    
    valid_anchor_box = np.array(valid_anchor_box)
    
    #print(valid_anchor_box.shape) #(4,8940,4)

    #tensor(4,8940)
    anchor_label = np.empty((batch_size,len(index_inside)), dtype = np.int32)
    anchor_label.fill(-1) #default other(-1)
    #print(anchor_label.shape)
    
    
    #IoU calc, IoU_list = ( )
    IoU_list = IoU(valid_anchor_box,area,bbox,batch_size)

    #sampling
    pos_iou_threshold = 0.7
    neg_iou_threshold = 0.3
    n_sample = 256
    

    for b in range(batch_size):
        for i in range(len(index_inside)):
            
            if IoU_list[b,i] >= pos_iou_threshold:
                anchor_label[b,i] = 1
            
            elif IoU_list[b,i] < neg_iou_threshold:
                anchor_label[b,i] = 0


    #mini_batch
    
    for b in range(batch_size):
        pos_index = np.where(anchor_label[b] == 1)[0]
        neg_index = np.where(anchor_label[b] == 0)[0]

        if len(pos_index) > n_sample/2:
            disable_idx = np.random.choice(pos_index,size = (len(pos_index)-(n_sample//2)),replace = False)

            anchor_label[b,disable_idx] = -1

        if len(neg_index) > n_sample/2:
            disable_idx = np.random.choice(neg_index, size = (len(neg_index)-(n_sample//2)),replace = False)
            anchor_label[b,disable_idx] = -1

        if len(pos_index) < n_sample/2:
            add_idx = np.random.choice(neg_index, size = ((n_sample//2)-len(pos_index)),replace = False)
            anchor_label[b,add_idx] = 1
    
    pos = anchor_label > 0
        
    '''
    for b in range(batch_size):
        pos_index = np.where(anchor_label[b] == 1)[0]
        neg_index = np.where(anchor_label[b] == 0)[0]
        print("개수",len(pos_index),len(neg_index), "index",pos_index,neg_index)
    '''
    return anchor_label,bbox_target,bbox_inside
    


#anchor_boxes = anchor_generator(4)

#forward(anchor_boxes,1,1,4)

