import torch
import torch.nn as nn
import torchvision

class CNN_pretrained(nn.Module):
    def __init__(self):
        super(CNN_pretrained,self).__init__()
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self. model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)

    def forward(self,x,train_box=None):
        if train_box is not None:
            x = self.model(x,train_box)
        else:
            x = self.model(x)
        return(x)