import torch.nn as nn
import torch



class CNN_class(nn.Module):
    def __init__(self,in_size,out_size):
        super(CNN_class, self).__init__()
        self.in_layer = nn.Conv2d(in_size,out_size//2,3,stride=2,padding=1)
        self.dropout1 = nn.Dropout(0.3)
        self.hd_layer = nn.Conv2d(out_size//2,out_size,3,stride=2,padding=1)
        
        self.dropout2 = nn.Dropout(0.3)
        # self.r_block = ResidualBlock(out_size,2)
        self.hd_layer2 = nn.Conv2d(out_size,out_size,3,stride=2,padding=1)
        self.dropout3 = nn.Dropout(0.3)
        self.hd_layer3 = nn.Conv2d(out_size,out_size,3,stride=2,padding=1)
        self.dropout4 = nn.Dropout(0.3)
        self.hd_layer4 = nn.Conv2d(out_size,out_size,3,stride=2,padding=1)

        # self.fc_out(32 *13 * 18,32 *13 * 18/2)

        # self.fc_out = nn.Linear(32 *14 * 18,512)
        # self.out = nn.Linear(512,10)

        self.num_anchors = 6  # Number of anchor boxes at each feature map location
        self.num_classes = 50
        # self.bbox_head = nn.Conv2d(out_size, self.num_anchors * 4, kernel_size=1)  # Bounding box regression
        # self.cls_head = nn.Conv2d(out_size, self.num_anchors * self.num_classes, kernel_size=1)  # Class pre
        self.cls_head = nn.Linear(2560,512)  # Class predictiondiction
        self.cls_head2 = nn.Linear(512,49)  # Class prediction
        self.bbx_head = nn.Linear(2560,512)  # Class predictiondiction
        self.bbx_head2 = nn.Linear(512,49)  # Class prediction
        
        
        

        
    def forward(self, x,is_train=True):
        h = (torch.relu(self.in_layer(x)))
        h = self.dropout1(h)
        h = torch.relu(self.hd_layer(h))
        h = self.dropout2(h)
        h = torch.relu(self.hd_layer2(h))
        h = self.dropout3(h)
        h = torch.relu(self.hd_layer3(h))
        h = self.dropout4(h)
        h = torch.relu(self.hd_layer4(h))
        # bbox = self.bbox_head(h)
        h = h.reshape(h.shape[0],-1)
        cls = torch.relu(self.cls_head(h))
        cls_one_hot =  torch.relu(self.cls_head2(cls))
        bbx = torch.relu(self.bbx_head(bbx))
        bbox =  torch.relu(self.bbx_head2(bbx))
        
        # bbox = bbox.permute(0,2,3,1).reshape(bbox.shape[0],-1,4)
        # cls = cls.permute(0,2,3,1).reshape(cls.shape[0],-1,self.num_classes)
        # cls = nn.functional.softmax(cls)
        
        # mask = torch.where(cls>0.3)
        # print(cls)
        # if len(mask[0])>0:
        #     print('here')
        #     cls = cls[mask]
        #     max_indices = torch.argmax(cls, dim=-1)
        #     print(cls.shape)
        #     cls_one_hot = torch.sum(nn.functional.one_hot(max_indices, num_classes=self.num_classes-1),axis=1)
        #     print(cls_one_hot.shape)
        # else:
        #     cls_one_hot = torch.zeros((cls.shape[0], self.num_classes-1),device=cls.device,requires_grad=True)

        return cls_one_hot,bbox
