import torch
import numpy as np


#input : gt bbox(4), target_anchor box(batch_size,anchor_idx,4)
def IoU(target_anchor,area,bbox,batch_size):

    IoU = np.empty((batch_size,len(target_anchor[0])),dtype=np.float32) #(4,8940)
    
    print("util.bbox",bbox)
    for b in range(batch_size):
        for i,anchor in enumerate(target_anchor[b]):
            xa1,ya1,xa2,ya2 = anchor #anchor box 좌표
            anchor_area = area
            #anchor_area = (xa2 - xa1) * (ya2 - ya1)

            xb1,yb1,xb2,yb2 = bbox
            box_area = (xb2 - xb1) * (yb2 - yb1)
            #print(bbox,xb1,yb1,xb2,yb2)

            #겹치는 box 좌하단, 우상단 x,y 좌표
            inter_x1 = max(xa1,xb1)
            inter_y1 = max(ya1,yb1)
            inter_x2 = min(xa2,xb2)
            inter_y2 = min(ya2,yb2)

            #iou calc
            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                iou = inter_area / (anchor_area + box_area - inter_area)

            else:
                iou = 0
    
            IoU[b,i] = iou

    return IoU

def bbox_transform(anchor,delta,batch_size):
    width = anchor[:,:,2] - anchor[:,:,0]
    height = anchor[:,:,3] - anchor[:,:,1]
    ctr_x = anchor[:,:,0] + 0.5 * width
    ctr_y = anchor[:,:,1] + 0.5 * height

    dx = delta[:,:,0::4]
    dy = delta[:,:,1::4]
    dw = delta[:,:,2::4]
    dh = delta[:,:,3::4]

    pred_ctr_x = dx * widht.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * height.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * width.unsqueeze(2)
    pred_h = torch.exp(dh) * height.unsqueeze(2)

    pred_box = delta.clone()

    # x1
    pred_box[:,:,0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_box[:,:,1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_box[:,:,2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_box[:,:,3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_box

def clip_box(box,img_shape, batch_size):

    for i in range(batch_size):
        box[i,:,0::4].clamp_(0, img_shape[i,1]-1)
        box[i,:,1::4].clamp_(0, img_shape[i,0]-1)
        box[i,:,2::4].clamp_(0, img_shape[i,1]-1)
        box[i,:,3::4].clamp_(0, img_shape[i,0]-1)

    return box

def nms(dets, thresh):
    
    dets = dets.numpy()
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    score = dets[:,4]

    area = (x2 -x1 + 1) * (y2- y2 + 1)
    order = score.argsort()[::-1]

    keep =[]

    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        
        inter = w * h
        ovr = inter / (area[i] + area[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return torch.IntTensor(keep)

def get_anchors(features,anchor,feat_stride=16):

    h,w = featurs.size()[-2:] #50,50

    shift_x = np.arrange(0,w) * feat_stride
    shift_y = np.arrange(0,h) * feat_stride

    shift_x, shift_y = np.meshgrid(shift_x,shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),shift_x.ravel(),shift_y.ravel())).transpose()
    
    A = anchor.shape[0] # 9
    K = shifts.shape[0] # 50 * 50

    all_anchors = (anchor.reshape((1,A,4)) + shifts.reshape((1,K,4)).transpose((1,0,2)))

    all_anchors = all_anchors.reshape((K*A, 4))


    return all_anchors
