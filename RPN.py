import torch
from torch.nn import functional as F
from anchor_generator import anchor_generator
from anchor_target_layer import anchor_target
from proposal_layer import proposal_layer
from util import get_anchors

class RPN(nn.Module):

    def __init__(self,in_channels):
        super(RPNhead,self).__init__()
        
        #input_feature map's depth (512)
        self.in_channels = in_channels

        self.intermediate = nn.Conv2d(self.in_channels, 512, kernel_size = 3, stride = 1, padding =1)

        self.cls = nn.Conv2d(512, 18, kernel_size = 1, stride = 1)
        
        self.reg = nn.Conv2d(512, 36, kernel_size = 1, stride = 1)

        self.anchor_generator = anchor_generator # load 해야함
        self.anchor_target = anchor_target

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    def reshape(x,d):
        input_shape = x.size()
        x = x.view(input_shape[0], int(d), int(float(input_shape[1] * input_shape[2]) / float(d)), input_shape[3])
        return x
    
    def forward(self, base_feat):
        
        #batch size check
        batch_size = base_feat.size(0)
        #feature = (batch_size,512,50,50)
        
        #(batch_size,512,50,50)
        rpn_conv1 = F.relu(self.intermediate(base_feat), inplace=True)
        
        #RPN classification score
        rpn_cls_score = self.cls(rpn_conv1) #(batch,18,50,50)->permute(batch,50,18,50)
        
        ## rpn score reshape & softmax
        score_reshape = self.reshape(rpn_cls_score, 2) # (batch, 2, (18*50)/2, 50)
        score_prob = F.softmax(score_reshape, 1)
        rpn_cls_score = self.reshape(score_prob, 18) # (batch, 18,50,50)
    
        #RPN bbox
        rpn_bbox_pred = self.reg(rpn_conv1) #(batch,36,50,50)        

        #proposal layer

        #anchor where???? 여기 들어가는 score가 위에서 softmax 적용하고 들어가야함
        anchor = self.anchor_generator #anchor가 이게 맞냐
        anchor = get_anchors(base_feat,anchor)

        #input(score:(batch,18,50,50), bbox:(batch,36,50,50), anchor:(50*50*9,4)
        rois = proposal_layer(rpn_cls_score,rpn_bbox_pred,anchor)

        print("rpn_cls_score",rpn_cls_score.shape) #(1,50,50,18)?
        print("rpn_bbox",rpn_bbox_pred.shape) #(1,18,50,50)?
        

        #RPN training -> rpn_label(minibatch) from anchor_target layer and rpn_loss

        if self.training:
            
            rpn_cls_score = rpn_cls_score.permute(0,2,3,1).contiguous().view(batch_size,-1,2) #(batch,22500,2)
            rpn_bbox_pred = rpn_bbox_pred.permute(0,2,3,1).contiguous().view(batch_size,-1,4) #(batch,22500,4)

            #get sampling anchor list rpn_label (batch_size,8940)
            rpn_label,bbox_target,bbox_inside  = self.anchor_target() 


            #compute classification loss            
            self.rpn_loss_cls = F.cress_entropy(rpn_cls_score[0], rpn_label, ignore_index = -1)
            print(rpn_loss_cls)


            #compute regression loss
            pos_sample = rpn_label > 0
            
            #positive gt box
            target_mask = valid_anchor_box[pos_sample].view(batch_size,-1,4) 
            #positive pred box
            pred_mask = rpn_bbox_pred[0][pos_sample].view(batch_size,-1,4) # 수정
            
            #diff pred and label
            x = torch.abs(target_mask.cpu() - pred_mask.cpu())
            self.rpn_loss_box = ((x<1).float() * 0.5 * x ** 2 ) + ((x>=1).float() * (x-0.5))
            
    return rois, rpn_loss_cls, rpn_loss_box
