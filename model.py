import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):

    def __init__(self):
        super(VGG,self).__init__()

        vggnet = models.vgg16(pretrained=True)
        modules = list(vggnet.children())[:-1]
        modules = list(modules[0])[:-1]

        self.vggnet = nn.Sequential(*modules)



    def forward(self,images):
        
        features = self.vggnet(images)
        
        return features

