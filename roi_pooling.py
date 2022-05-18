import torch
import torch.nn as nn

class ROIpooling(nn.Module):

    def __init__(self, size=(7, 7), spatial_scale=1.0 / 16.0): 
        super(ROIpooling, self).__init__()
        
        self.adapmax2d = nn.AdaptiveMaxPool2d(size)
        self.spatial_scale = spatial_scale
   
    def forward(self, features, rois_boxes)         
        
        # rois_boxes : [x, y, x`, y`]
        rois_boxes = rois_boxes.data.float().clone()
        #roi box 크기를 down sampling 하여 feature 크기에 맞춰줌
        rois_boxes.mul_(self.spatial_scale) 
        rois_boxes = rois_boxes.long()
        output = []
        
        for i in range(rois_boxes.size(0)):                 
            roi = rois_boxes[i]                                     
            
            #roi 영역 크롭 feature(batch,512,50,50)?
            roi_feature = features[:, :, roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]
            
            #자른 영역에서 max pooling                      
            pool_feature = self.adapmax2d(roi_feature) #(batch,512,7,7)??   
            output.append(pool_feature)                                     
        
        return torch.cat(output, 0)#tensor 리스트를 한번에 tensor로 만들어줌
        #faster rcnn에 입력되는 roi feature는 입력채널이 512*7*7임
