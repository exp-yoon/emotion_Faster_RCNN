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
from RPN import RPN
import pandas as pd
from emotic_dataset import Emotic_CSVDataset
#import fasterRCNN

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
            transforms.Resize((800,800)),
                    transforms.ToTensor()]) 

    batch_size = 2
    '''
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
    '''

    context_mean = [0.4690646, 0.4407227, 0.40508908]
    context_std = [0.2514227, 0.24312855, 0.24266963]
    body_mean = [0.43832874, 0.3964344, 0.3706214]
    body_std = [0.24784276, 0.23621225, 0.2323653]
    context_norm = [context_mean, context_std]
    body_norm = [body_mean, body_std]

    cat2ind = {}

    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection','Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear','Happiness','Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

    for idx,emotion in enumerate(cat):
        cat2ind[emotion] = idx

    data_df = pd.read_csv('./data/emotic_pre/train.csv')
    train_dataset = Emotic_CSVDataset(data_df,cat2ind,transform,context_norm,body_norm,data_src='./emotic')

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
    model = VGG()
    rpn = RPN()
    data_iter = iter(train_loader)
    image,bbox,label,area = next(data_iter)
    
    for i, data in enumerate(train_loader,0):
        
        inputs, bbox,label,area  = data
        
        #tensor(N,3,800,800)
        output = model(inputs)
        
        out1,out2,out3=rpn(output,bbox,area)
    '''
    
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
    '''
    dataiter = iter(train_loader)
    images,bbox,label = dataiter.next()
    
    
    img = images[0].numpy()
    plt.imshow(np.transpose(img,(1,2,0)))
    plt.show()
    '''

    #targets = [{k: v for k,v in t.items()} for t in targets]
    #print(targets)
        

    '''
    for images,targets in train_load:
        images = list(image for image in images)
        targets = [{k: v for k,v in t.items()} for t in targets]
        print(images)
        print(targets)
    '''

main()


