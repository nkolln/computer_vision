import torch.nn as nn
import torch



class CNN_class(nn.Module):
    def __init__(self,in_size,out_size,d_out=256,hidden_dim=512,beta=1,jitter=False):
        super(CNN_class, self).__init__()
        self.in_layer = nn.Conv2d(in_size,out_size//2,3,stride=2,padding=1)
        self.dropout1 = nn.Dropout(0.3)
        self.hd_layer = nn.Conv2d(out_size//2,out_size,3,stride=2,padding=1)
        
        self.dropout2 = nn.Dropout(0.3)
        # self.r_block = ResidualBlock(out_size,2)
        self.hd_layer2 = nn.Conv2d(out_size,out_size,3,stride=2,padding=1)
        self.dropout3 = nn.Dropout(0.3)
        self.hd_layer3 = nn.Conv2d(out_size,out_size,3,stride=2,padding=1)

        # self.fc_out(32 *13 * 18,32 *13 * 18/2)

        self.fc_out = nn.Linear(32 *14 * 18,512)
        self.out = nn.Linear(512,10)
        

        
    def forward(self, x,is_train=True):
        print(x.shape)
        h = (torch.relu(self.in_layer(x)))
        h = self.dropout1(h)
        h = torch.relu(self.hd_layer(h))
        h = self.dropout2(h)
        h = torch.relu(self.hd_layer2(h))
        h = self.dropout3(h)
        h = torch.relu(self.hd_layer3(h))
        # h = self.dropout4(h)
        # h = self.r_block(h)

        h = h.reshape(h.shape[0],-1)
        print(h.shape)
        h = torch.relu(self.fc_out(h))
        h = torch.sigmoid(self.out(h))

        # #continuous
        # vq_loss = kl_loss(p,c)
        # quantized_inputs = p

        return h
