import torch.nn as nn
import torch
from torch.autograd import Variable


def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu
    
class Residual(nn.Module):
    def __init__(self,l_in):
        super(Residual,self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(False),
            nn.Conv2d(l_in,l_in*4,[1,3],stride=[1,1],padding=[0,1],bias=False),
            nn.ReLU(False),
            nn.Conv2d(l_in*4,l_in,[1,1],stride=[1,1],padding=[0,0],bias=False),
            )
    
    def forward(self,x):
        return(x+self.block(x))

class ResidualBlock(nn.Module):
    def __init__(self,in_size,num_layer):
        super(ResidualBlock,self).__init__()
        self.layers = nn.ModuleList([Residual(in_size) for _ in range(num_layer)])
        self.num_layer = num_layer

    def forward(self,x):
        for i in range(self.num_layer):
            x = self.layers[i](x)
        x = torch.nn.functional.relu(x)
        return(x)

class Encoder_conv(nn.Module):
    def __init__(self,in_size,out_size,d_out=256,hidden_dim=512,beta=1,jitter=False):
        super(Encoder_conv, self).__init__()
        self.in_layer = nn.Conv2d(in_size,out_size//2,3,stride=2,padding=1)
        self.hd_layer = nn.Conv2d(out_size//2,out_size,3,stride=2,padding=1)
        self.hd_layer2 = nn.Conv2d(out_size,out_size,3,stride=2,padding=1)
        self.hd_layer3 = nn.Conv2d(out_size,out_size,3,stride=2,padding=1)

        # self.fc_out(32 *13 * 18,32 *13 * 18/2)

        self.mu_layer = nn.Linear(32 *14 * 18,512)
        self.sp_layer = nn.Linear(32 *14 * 18,512)
        

        
    def forward(self, x,is_train=True):
        h = torch.relu(self.in_layer(x))
        h = torch.relu(self.hd_layer(h))
        h = torch.relu(self.hd_layer2(h))
        h = torch.relu(self.hd_layer3(h))

        h = h.reshape(h.shape[0],-1)
        mu = self.mu_layer(h)
        sp = self.sp_layer(h)

        z_sample = reparameterize(is_train,mu,sp)
        
        # #continuous
        # vq_loss = kl_loss(p,c)
        # quantized_inputs = p

        return z_sample,mu, sp

class Decoder_conv(nn.Module):
    def __init__(self,in_size,out_size,hidden_dim=None,mod=0):
        super(Decoder_conv, self).__init__()
        self.in_layer_lin = nn.Linear(512,32 *14 * 18)
        self.hd_layer = nn.ConvTranspose2d(in_size,in_size,3, stride=2, padding=1, output_padding=[0,1])
        self.hd_layer1 = nn.ConvTranspose2d(in_size,in_size,3, stride=2, padding=1, output_padding=[1,1])
        self.hd_layer2 = nn.ConvTranspose2d(in_size,in_size//2,3, stride=2, padding=1, output_padding=[1,1])
        self.hd_layer3 = nn.ConvTranspose2d(in_size//2,out_size,3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        h = torch.relu(self.in_layer_lin(x).reshape(x.shape[0],32,14,18))
        h = torch.relu(self.hd_layer(h))
        
        h = torch.relu(self.hd_layer1(h))
        h = torch.relu(self.hd_layer2(h))
        x_reconstr = torch.sigmoid(self.hd_layer3(h))
        
        return x_reconstr
    

class Encoder(nn.Module):
    def __init__(self,in_size,out_size,d_out=256,hidden_dim=512,beta=1,jitter=False):
        super(Encoder, self).__init__()
        self.in_layer = nn.Linear(in_size,hidden_dim//2)
        self.hd_layer = nn.Linear(hidden_dim//2,hidden_dim)
        self.hd_layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.out_size = out_size
        self.mu_layer = nn.Linear(hidden_dim,512)
        self.sp_layer = nn.Linear(hidden_dim,512)

        

        
    def forward(self, x,is_train=True):
        x = x.float()
        h = torch.relu(self.in_layer(x))
        h = torch.relu(self.hd_layer(h))
        h = torch.relu(self.hd_layer2(h))
        mu = self.mu_layer(h)
        sp = self.sp_layer(h)

        z_sample = reparameterize(is_train,mu,sp)
        
        # #continuous
        # vq_loss = kl_loss(p,c)
        # quantized_inputs = p

        return z_sample,mu, sp

class Decoder(nn.Module):
    def __init__(self,in_size,out_size,d_out=256,hidden_dim=512):
        super(Decoder, self).__init__()
        self.in_layer = nn.Linear(512,hidden_dim)
        self.hd_layer = nn.Linear(hidden_dim,hidden_dim//2)
        self.hd_layer2 = nn.Linear(hidden_dim//2,in_size)
        self.out_layer = nn.Linear(in_size,in_size)
        
    def forward(self, x):
        x = x.reshape(*x.shape[:1],-1)
        h = torch.relu(self.in_layer(x))
        h = torch.relu(self.hd_layer(h))
        h = torch.relu(self.hd_layer2(h))
        x_reconstr = torch.sigmoid(self.out_layer(h))
        return x_reconstr