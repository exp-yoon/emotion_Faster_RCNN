import torch
import torch.nn as nn
import torch.nn.functional as F

import roi_pooling


class fasterRCNN(nn.Module):

    def __init__(self,class_num):
        super(fasterRCNN,self).__init__()

        self.class_num = class_num
        
        self.fc1 = nn.Sequential(nn.Linear(512*7*7, 4096),
                                 nn.ReLU(),
                                 nn.Dropout())

        self.fc2 = nn.Sequential(nn.Linear(4096,4096),
                                 nn.ReLU(),
                                 nn.Dropout())

        #class num + background(1)
        self.classfier = nn.Linear(4096,class_num + 1)
        self.softmax = nn.Softmax()

        self.regressor = nn.Linear(4096, (class_num + 1) * 4)


        #loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        
        self.proposal_target = proposal_target()
        self.ROI_pool = roi_pooling()

    def forward(self,features,proposal,im_info,gt_box):

        #proposal from rpn's proposal_layer, 
        #for faster rcnn training
        roi_label, rois, bbox_target, bbox_inside = proposal_target(proposal,gt_box ) 
        

        if self.training:
            
            #feature from vgg16, rois from proposal_target_layer
            #roi pooling layer
            pooled_feature = roi_pooling(features, rois)
            
            pooled_feature = pooled_feature.view(-1,512*7*7)
            
            pooled_feature = self.fc1(pooled_feature)
            
            pooled_feature = self.fc2(pooled_feature)

            #logit ??
            logits = self.classfier(pooled_feature)
            score = self.softmax(logits)
            bbox_delta = self.regressor(pooled_feature)



    return rcnn_cls_loss, rcnn_reg_loss
