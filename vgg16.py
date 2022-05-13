import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG(nn.Module):


    def _init_modules(self):
        vgg = models.vgg16(pretrained=True).to(device)
        
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

