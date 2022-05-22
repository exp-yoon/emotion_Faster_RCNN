import torch
import torch.nn as nn
from torch.nn import functional as F
from anchor_generator import anchor_generator
from anchor_target_layer import anchor_target
from proposal_layer import proposal_layer
#from util import get_anchors
from util import smooth_l1_loss
from torch.autograd import Variable

def reshape(x,d):
    input_shape = x.size()
    x = x.view(input_shape[0],int(d),int(float(input_shape[1]*input_shape[2])/float(d)),input_shape[3])
    return x

class RPN(nn.Module):

    def __init__(self):
        super(RPN,self).__init__()
        
        #input_feature map's depth (512)
        self.in_channels = 512

        self.intermediate = nn.Conv2d(self.in_channels, 512, kernel_size = 3, stride = 1, padding =1)

        self.cls = nn.Conv2d(512, 18, kernel_size = 1, stride = 1)
        
        self.reg = nn.Conv2d(512, 36, kernel_size = 1, stride = 1)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    
    def forward(self, base_feat,bbox,area):
        
        #batch size check
        batch_size = base_feat.size(0)
        #feature = (batch_size,512,50,50)
        
        #(batch_size,512,50,50)
        rpn_conv1 = F.relu(self.intermediate(base_feat), inplace=True)
        
        #RPN classification score
        rpn_cls_score = self.cls(rpn_conv1) #(batch,18,50,50)
                
        ## rpn score reshape & softmax
        score_reshape = reshape(rpn_cls_score,2) # (batch, 2, (18*50)/2, 50)
        score_prob = F.softmax(score_reshape, 1)
        rpn_cls_score = reshape(score_prob, 18) # (batch, 18,50,50)
    
        #RPN bbox
        rpn_bbox_pred = self.reg(rpn_conv1) #(batch,36,50,50)        

        #proposal layer

        anchor = anchor_generator() #(22500,4)
        anchor = torch.from_numpy(anchor)
    
        #input(score:(batch,18,50,50), bbox:(batch,36,50,50), anchor:(22500,4)
        rois = proposal_layer(rpn_cls_score,rpn_bbox_pred,anchor)
        #rois : (N,2000,5)

        #RPN training -> rpn_label(minibatch) from anchor_target layer and rpn_loss

        if self.training:
            
            #(N,22500,2)
            rpn_cls_score = rpn_cls_score.permute(0,2,3,1).contiguous().view(batch_size,-1,2) 

            #get sampling anchor list rpn_label (batch_size,8940)
            #bbox_target :(N,36,50,50), inside_weight : (N,36,50,50)
            rpn_label,bbox_target,inside_weight  = anchor_target(anchor,bbox,area,batch_size) 
            rpn_label = rpn_label.view(batch_size,-1)#(N,22500)
            
            #rpn_keep : (512) -> minibatch 256 * N
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            #rpn_cls_score = (512,2)
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2),0,rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1),0,rpn_keep)
            rpn_label = Variable(rpn_label.long())
        
            #compute classification loss            
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label, ignore_index = -1)
            print("cls_loss",self.rpn_loss_cls)

            #compute regression loss
            inside_weight = Variable(inside_weight)
            bbox_target = Variable(bbox_target)

            self.rpn_loss_box = smooth_l1_loss(rpn_bbox_pred, bbox_target,inside_weight,sigma=3,dim=[1,2,3])

            print("reg_loss",self.rpn_loss_box)
            
            print("roi",rois,rois.shape)
        return rois, self.rpn_loss_cls, self.rpn_loss_box
