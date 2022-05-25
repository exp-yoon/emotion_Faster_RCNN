import torch
import torch.nn as nn
import numpy as np
from util import IoU_roi


def proposal_target_layer(roi,gt_box,area,batch_size):
    
    num_class = 26
    n_sample = 128
    iou_threshold = 0.5
    pos_ratio = 0.25
    neg_ratio = 0.75
    
    gt_box = gt_box.view(batch_size,1,-1)
    gt_box_append = gt_box.new(gt_box.size()).zero_()
    #fisrt bbox coord, 1box per image -> (N,1,5)
    gt_box_append[:,:,1:5] = gt_box[:,:,:4]

    #all_roi : (N,1000,5), gt_box_append : (N,1,5) -> CAT (N,1000+1,5)
    all_roi = torch.cat([roi,gt_box_append],1)
    

    num_image = 1
    roi_per_image = int(batch_size/num_image) #N
    #minubatch pos ratio -> N*0.25
    fg_roi_per_image = int(np.round(pos_ratio * roi_per_image))
    fg_roi_per_image = 1 if fg_roi_per_image == 0 else fg_roi_per_image 

    labels,rois,bbox_targets,bbox_inside_weight = sample_rois(all_roi,area,gt_box,fg_roi_per_image,roi_per_image,num_class)

    return rois,label,bbox_targets,bbox_inside_weight

def sample_rois(all_roi,area,gt_box,fg_roi_per_image,roi_per_image,num_class):
    gt_box = gt_box.squeeze(1)[:,:4]
    all_roi = all_roi[:,:,1:5]
    print(all_roi.shape)
    batch_size = 2
    #calc iou : (N,num_roi(1000),1)    

    IoU_list = IoU_roi(all_roi,area,gt_box,batch_size)
    IoU_list = torch.from_numpy(IoU_list)
    print(IoU_list,IoU_list.shape)
    max_iou, max_iou_idx = torch.max(IoU_list,1)
    print(max_iou,max_iou_idx)

    batch_size = IoU_list.size(0)
    print(batch_size)
    #arrange -> tensor(0,1..N-1) : (N) , gt_box.size(1) = 1
    offset = torch.arrange(0,batch_size) * gt_box.size(1)
    #offset : (N,1) + max roi idx 값 더하기
    offset = offset.view(-1,1) + max_iou_idx

    #label : ()
    labels = gt_box[:,:,4].contiguous().view(-1).index((offset.view(-1),)).view(batch_size,-1)
