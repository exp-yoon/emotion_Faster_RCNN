import numpy as np
from util import bbox_transform
from util import clip_box
from util import nms
import torch


#proposal layer : extract region proposal, NMS  

def proposal_layer(score,bbox_delta,anchor):
    #score(batch,18,50,50) delta(batch,36,50,50)
    #anchor(22500,4)
    img_shape = (800,800)

    batch_size = score.size(0)
        
    #anchor : (N,22500,4)
    anchor = anchor.expand(batch_size,22500,4)
    
    #bbox_delta : (N,22500,4)
    bbox_delta = bbox_delta.permute(0,2,3,1).contiguous()
    bbox_delta = bbox_delta.view(batch_size,-1,4)

    #bbox transform (N,22500,4)
    proposal_boxes = bbox_transform(anchor,bbox_delta,batch_size)

    #score : (N,9,50,50)
    score = score[:,9:,:,:]
    
    #score : (N,22500)
    score = score.permute(0,2,3,1).contiguous()
    score = score.view(batch_size, -1)
    
    #clip predicted box to image
    proposal = clip_box(proposal_boxes,img_shape, batch_size) #img_shape 정의

    nms_thresh = 0.7
    n_pre_nms = 12000
    n_post_nms = 2000
    #sorted score list DESC
    #order : (2,22500) -> 높은 score순서대로 anchor의 index 반환
    _,order = torch.sort(score, 1,True)
    #select top N

    score_keep = score
    proposal_keep = proposal
    output = score.new(batch_size, n_post_nms, 5).zero_()
    
    for i in range(batch_size):
        
        proposal_single = proposal_keep[i]
        score_single = score_keep[i]

        order_single = order[i]

        #order_single : tensor (12000)
        if n_pre_nms > 0 and n_pre_nms < score.numel():
            order_single = order_single[:n_pre_nms]

        proposal_single = proposal_single[order_single,:]
        score_single = score_single[order_single].view(-1,1)

        keep_idx = nms(torch.cat((proposal_single,score_single),1),nms_thresh)
        keep_idx = keep_idx.long().view(-1)

        if n_post_nms > 0:
            keep_idx = keep_idx[:n_post_nms]

        proposal_single = proposal_single[keep_idx, :]
        score_single = score_single[keep_idx, :]

        num_proposal = proposal_single.size(0)
        output[i,:,0] = i
        output[i, :num_proposal, 1:] = proposal_single

    return output 
