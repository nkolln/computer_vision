import torch
import torch.nn as nn
import torchvision
import os

class CNN_pretrained(nn.Module):
    def __init__(self,variant):
        super(CNN_pretrained,self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        if variant['new_load']:
            self.model = torch.load('models/cnn_fine_tuned.pt')

    def forward(self,x,train_box=None):
        if train_box is not None:
            x = self.model(x,train_box)
        else:
            x = self.model(x)
        return(x)