import numpy as np
from util import bbox_transform
from util import clip_box
from util import get_anchors

#proposal layer : extract region proposal, NMS  

def proposal_layer(score,bbox_delta,anchor):
    #score(batch,18,50,50) delta(batch,36,50,50)
    #anchor(2500*9,4)

    anchor = get_anchors(anchor)

    #bbox_transform
    bbox_delta = bbox_delta.permute(0,2,3,1).contiguous()
    bbox_delta = bbox_delta.view(batch_size,-1,4)

    #bbox transform
    proposal_boxes = bbox_transform(anchor,bbox_delta)
    
    score = score.permute(0,2,3,1).contiguous()
    score = score.view(batch_size, -1)

    proposal = bbox_transform(anchor,bbox_delta,batch_size)


    ### check here

    #clip predicted box to image
    proposal = clip_box(proposal,img_shape, batch_size) #img_shape 정의
    
    nms_thresh = 0.7
    n_pre_nms = 12000
    n_post_nms = 2000
    #sorted score list DESC
    order = torch.sort(score, 1,True)
    #select top N

    output = score.new(batch_size, n_post_nms, 5).zero_()

    for i in range(batch_size):
        
        proposal_single = proposal[i]
        score_single = score[i]

        order_single = order[i]

        if n_pre_nms > 0 and n_pre_nms < score.numel():
            order_single = order_single[:n_pre_nms]

        proposal_single = proposal_single[order_single,:]
        score_single = score_single[order_single].view(-1,1)

        keep_idx = nms(torch.cat((proposal_single,score_single),1),nms_thresh)
        keep_idx = keep.idx.long().view(-1)

        if n_post_nms > 0:
            keep_idx = keep_idx[:n_post_nms]

        proposal_single = proposal_single[keep_idx, :]
        score_single = score_single[keep_idx, :]

        num_proposal = proposal single.size(0)
        output[i,:,0] = i
        output[i, :num_proposal, 1:] = proposal_single

    return output 
