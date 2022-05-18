import torch
import numpy as np
from anchor_generator import anchor_generator
from util import IoU
from util import bbox_transform_batch
#inside anchor box select, calc IoU valid anchor & gt bbox
#sampling anchor box positive(>=0.7) , negative(<= 0.3), other(-1)

def anchor_target(anchor, bbox, area ,batch_size):
    '''
    anchor : (22500,4)
    gt_box : (N,4)

    anchor_label : (N,22500)
    bbox_target : (N,22500,4)
    bbox_inside : (N,22500,4)
    
    '''
    

    anchor_box = anchor # (22500,4)
    total_anchor = len(anchor_box)
    
    #valid anchor box select, img size->input
    index_inside = ((anchor_box[:,0] >= 0) &
            (anchor_box[:,1] >= 0) &
            (anchor_box[:,2] <= 800) &
            (anchor_box[:,3] <= 800))
    
    #return inside anchor index
    inside_anchor = np.where(index_inside == 1)   
    

    #anchors : (8940,4)
    anchors = anchor_box[inside_anchor,:].squeeze(0)
    

    #anchor_label : (N,8940)
    anchor_label = np.empty((batch_size,len(inside_anchor[0])), dtype = np.int32)
    anchor_label.fill(-1) #default other(-1)
       
    
    #IoU calc, IoU_list = (N,8940)
    IoU_list = IoU(anchors,area,bbox,batch_size)
    

    #sampling
    pos_iou_threshold = 0.7
    neg_iou_threshold = 0.3
    n_sample = 256
    

    for b in range(batch_size):
        for i in range(len(anchors)):
            
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

    #anchor shape expand(8940,4)->(N,8940,4)
    batch_anchors = torch.from_numpy(anchors)
    batch_anchors = batch_anchors.expand(batch_size,len(anchors),4)
    
    #gt_box shape expand(N,4)->(N,8940,4)
    gt_box = torch.from_numpy(bbox)
    gt_box = gt_box.repeat(1,8940).view(batch_size,8940,4)
    
    #anchors : (N,8940,4) , gt_box : (N,8940,4)
    bbox_target = bbox_transform_batch(batch_anchors,gt_box)
    
    # (N,8940,4)
    bbox_inside_weight = np.zeros((batch_size,len(inside_anchor[0]),4)) 

    bbox_inside_weight[anchor_label == 1] = [1.0,1.0,1.0,1.0]
    
    bbox_inside_weight = torch.from_numpy(bbox_inside_weight)
    
    # transform to tensor
    anchor_label = torch.from_numpy(anchor_label)
    inside_anchor = torch.from_numpy(inside_anchor[0])

    #8940 -> 22500
    #anchor_label:(2,22500) , bbox_target : (2,22500,4), bbox_inside_weight : (2,22500,4)
    anchor_label = _unmap(anchor_label,total_anchor,inside_anchor,batch_size,fill=-1)
    bbox_target = _unmap(bbox_target,total_anchor,inside_anchor,batch_size,fill=0)
    bbox_inside_weight = _unmap(bbox_inside_weight,total_anchor,inside_anchor,batch_size,fill=0)


    anchor_label = anchor_label.view(batch_size,50,50,9).permute(0,3,1,2).contiguous()
    anchor_label = anchor_label.view(batch_size,1,9*50*50) #(N,1,22500)
    ## 근데 여기 왜 (N,1,50,9*50)으로 하지..
    
    bbox_target = bbox_target.view(batch_size,50,50,36).permute(0,3,1,2).contiguous()

    anchor_count = bbox_inside_weight.size(1) # 22500
    bbox_inside_weight = bbox_inside_weight.view(batch_size,anchor_count,4)
    bbox_inside_weight = bbox_inside_weight.contiguous().view(batch_size,50,50,36).permute(0,3,1,2).contiguous()

    #(N,1,22500) , (N,36,50,50), (N,36,50,50)
    return anchor_label,bbox_target,bbox_inside_weight
    

def _unmap(data,count,inds,batch_size,fill=0):
    #fill data in inside index, else 0
    #data : label () or bbox_target ()
    #count : number of anchor (22500), inds = inside index
    #ret : (N,22500) or (N,22500,4)

    if data.ndim == 2:
        # data = label
        ret = torch.Tensor(batch_size,count).fill_(fill).type(torch.int32) #(N,22500)
        ret[:,inds] = data  

    else:
        #data = bbox_target or bbox_inside_weight
        ret = torch.Tensor(batch_size,count,data.size(2)).fill_(fill).type(torch.float64)
        ret[:,inds,:] = data

    return ret


anchor_boxes = anchor_generator(2)

gt_box = np.array([[10,50,40,80],[20,60,40,100]])

anchor_target(anchor_boxes,gt_box,[20,50],2)

