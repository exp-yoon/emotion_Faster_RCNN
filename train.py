import os
import torch
import torchvision
from torch.utils.data import DataLoader 
from torchvision import transforms
from coco_dataset import coco
import matplotlib.pyplot as plt
import numpy as np
import cv2
from anchor_generator import anchor_generator
from anchor_target_layer import anchor_target
from model import VGG
import RPN
import fasterRCNN

def main():
    devide = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
            transforms.Resize((800,800)),
                    transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) 

    batch_size = 3

    train_set = coco('./coco','train','2017', transform)

    train_load = torch.utils.data.DataLoader(train_set,batch_size=batch_size ,shuffle=True,collate_fn=train_set.collate_fn)


    data_iter = iter(train_load)
    images,targets = next(data_iter) 
    bbox = targets[0]['boxes']
    label = targets[0]['labels']
    image_id = targets[0]['image_id']
    area = targets[0]['area']
    is_crowd = targets[0]['iscrowd']

    num_objs = len(bbox)
    #print("train.bbox",bbox)
    if num_objs != 1 :
        bbox = bbox[0]
        label = label[0]
        area = area[0]
        is_crowd = is_crowd[0]

    #print(bbox,label,image_id,area,is_crowd)

    #anchor = anchor_generator(batch_size)
    #valid = anchor_target(anchor,bbox,area,batch_size)
    #print("anchor",anchor.shape,"valid",valid.shape)

    model = VGG()
    #print(data)
    #print("images",images)
    #print("targets",targets)    

    for i, data in enumerate(train_load,0):
        
        inputs, label = data
        print(inputs[0].unsqueeze(0))
        #tensor(1,3,800,800)
        output = model(inputs[0].unsqueeze(0))



    for epoch in range(epochs):

        for i,data in enumerate(train_load,0):
            
            images,targets = data
            
            #RPN & fasterrcnn init check

            features = VGG(images)
            proposal, rpn_loss_cls, rpn_loss_reg = RPN.forward(features)
            rcnn_loss_cls, rcnn_loss_reg = fasterRCNN.forward(features,proposal,im_info,gt_box)
            #backward

            rpn_loss = rpn_loss_cls + rpn_loss_reg
            rcnn_loss = rcnn_loss_cls + rcnn_loss_Reg
            loss = rpn_loss + rcnn_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    '''
    dataiter = iter(train_load)
    images, targets = dataiter.next()
    
    
    img = images[0].numpy()
    plt.imshow(np.transpose(img,(1,2,0)))
    plt.show()

    targets = [{k: v for k,v in t.items()} for t in targets]
    print(targets)
    '''    

    '''
    for images,targets in train_load:
        images = list(image for image in images)
        targets = [{k: v for k,v in t.items()} for t in targets]
        print(images)
        print(targets)
    '''

main()


