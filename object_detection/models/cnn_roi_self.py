import torch.nn as nn
import torch
import torchvision
from torchvision.ops import roi_pool


class CNN_roi(nn.Module):
    def __init__(self,in_channels,out_channels,num_classes=49):
        super(CNN_roi,self).__init__()
        self.backbone = nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-2]) 
        self.roi_pool = roi_pool
        self.cls_head = nn.Linear(self.out_channels * 7 * 7, num_classes)
        self.bbx_head = nn.Linear(self.out_channels * 7 * 7, num_classes*4)
