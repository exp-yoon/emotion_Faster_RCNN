import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
def anchor_generator():

    #img shape check
    #image_size = images.shape  
    ratio = [0.5,1,2]
    scale = [8,16,32]
    sub_sample = 16

    #feature size check
    feature_size = 50
    anchor_num = 9*(feature_size**2)

    #ex) img = 224, feature = 50, batch_size = 16

    #anchor box center, tensor(batch_size, anchor's num, 2) 2:x,y
    grid = np.zeros((feature_size**2,2))

    #grid step x,y coord on original image, start:sub_sample stop:(feature_size+1) step:sub_sample
    grid_x = np.arange(sub_sample,(feature_size+1)*sub_sample,sub_sample)
    grid_y = np.arange(sub_sample,(feature_size+1)*sub_sample,sub_sample)

    #tensor(anchor's num , 4)
    anchor_box = np.zeros((anchor_num,4)) # anchor box x1,y1,x2,y2
    #anchor box center(grid_cell) 
    
    
    # ex) grid_x,y = 16 -> grid = (8,8) ->anchor box center 
    index = 0
    for i in range(len(grid_y)):
        for j in range(len(grid_x)):
            grid[index,0] = grid_x[j] - (sub_sample/2)
            grid[index,1] = grid_y[i] - (sub_sample/2)
            index += 1
    '''
    img = cv2.imread("./coco/train2017/000000581929.jpg")
    img = cv2.resize(img,dsize=(800,800),interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(img,400,400,400,400,cv2.BORDER_CONSTANT, value=(255,255,255))
    '''

    index = 0
    for c in grid:
        c_x,c_y = c

        for i in range(len(ratio)):
            for j in range(len(scale)):

                #anchor box h,w
                h = sub_sample * scale[j] * np.sqrt(ratio[i])
                w = sub_sample * scale[j] * np.sqrt(1./ratio[i])

                anchor_box[index,0] = c_x - w / 2
                anchor_box[index,1] = c_y - h / 2
                anchor_box[index,2] = c_x + w / 2
                anchor_box[index,3] = c_y + h / 2
                index += 1
    '''
    for i in range(len(anchor_box[0])):
        x1 = int(anchor_box[0][i][0])
        y1 = int(anchor_box[0][i][1])
        x2 = int(anchor_box[0][i][2])
        y2 = int(anchor_box[0][i][3])
        cv2.rectangle(img,(x1+400,y1+400),(x2+400,y2+400),color=(255,0,0),thickness = 3)

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()
    '''
    #anchor_box = torch.from_numpy(anchor_box)
    return anchor_box   

#anchor_generator(16)
