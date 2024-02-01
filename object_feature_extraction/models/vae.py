



import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical

import torch
from torch import nn
from torch.nn import functional as F

class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25,
                 jitter = False):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

        self.jitter = jitter

    def forward(self, latents):
        
        latents = latents.permute(0, 2, 3, 1).contiguous()
        input_shape = latents.shape
        
        # Flatten input
        flat_input = latents.view(-1, self.D)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.K, device=latents.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), latents)
        q_latent_loss = F.mse_loss(quantized, latents.detach())
        loss = q_latent_loss + self.beta * e_latent_loss
        
        quantized = latents + (quantized - latents).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings




import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self,l_in):
        super(Residual,self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(l_in,l_in*4,[1,3],stride=[1,1],padding=[0,1],bias=False),
            nn.ReLU(True),
            nn.Conv2d(l_in*4,l_in,[1,1],stride=[1,1],padding=[0,0],bias=False),
            )
    
    def forward(self,x):
        return(x+self.block(x))

class ResidualBlock(nn.Module):
    def __init__(self,l_in,num_layer_r):
        super(ResidualBlock,self).__init__()
        self.layers = nn.ModuleList([Residual(l_in) for _ in range(num_layer_r)])
        self.num_layer_r = num_layer_r

    def forward(self,x):
        for i in range(self.num_layer_r):
            x = self.layers[i](x)
        x = F.relu(x)
        return(x)

class Encoder_conv(nn.Module):
    def __init__(self,l_in,l_out,hidden_dim,d_out,beta=1,jitter=False):
        super(Encoder_conv, self).__init__()
        self.temperature = 1
        self.in_layer = nn.Conv2d(l_in,l_out//2,[1,4],stride=[1,2],padding=[0,1])# Linear(l_in,hidden_dim)
        self.hd_layer = nn.Conv2d(l_out//2,l_out,[1,4],stride=[1,2],padding=[0,1])
        self.hd_layer2 = nn.Conv2d(l_out,l_out,[1,3],stride=[1,1],padding=[0,1])
        self.r_block = ResidualBlock(l_out,2) #set num layer hidden to 2 for now
        self.mu_layer = nn.Linear(hidden_dim,hidden_dim)
        self.c_layer = nn.Linear(hidden_dim,hidden_dim)
        self.emb_dim = l_out
        self.num_embed = d_out
        self.vq_layer = VectorQuantizer(self.num_embed,
                                        self.emb_dim,
                                        beta,
                                        jitter)

        
    def forward(self, x):
        x_in = x.reshape(x.shape[0],1,x.shape[1],x.shape[2])
        h = torch.relu(self.in_layer(x_in))
        h = torch.relu(self.hd_layer(h))
        h = torch.relu(self.hd_layer2(h))
        h = self.r_block(h)
        h_shape = h.shape
        h = h.reshape(x.shape[0],x.shape[1],-1)
        p = self.mu_layer(h)
        p = p.reshape(h_shape)
        c = self.c_layer(h)

        # discrete
        vq_loss,quantized_inputs,perplexity, encodings = self.vq_layer(p)
        quantized_inputs = quantized_inputs.reshape(*h.shape[:2],-1)
        
        # #continuous
        # vq_loss = kl_loss(p,c)
        # quantized_inputs = p

        return quantized_inputs, vq_loss, c
    
class Encoder(nn.Module):
    def __init__(self,l_in,l_out,hidden_dim,d_out,beta=1,jitter=False):
        super(Encoder, self).__init__()
        self.temperature = 1
        self.in_layer = nn.Conv2d(l_in,l_out*2,[1,4],stride=[1,2],padding=[0,1])# Linear(l_in,hidden_dim)
        self.hd_layer = nn.Conv2d(l_out*2,l_out*4,[1,4],stride=[1,2],padding=[0,1])
        self.hd_layer2 = nn.Conv2d(l_out*4,l_out*4,[1,3],stride=[1,1],padding=[0,1])
        self.r_block = ResidualBlock(l_out*4,2) #set num layer hidden to 2 for now
        self.mu_layer = nn.Linear(hidden_dim,hidden_dim)
        self.c_layer = nn.Linear(hidden_dim,hidden_dim)
        self.emb_dim = l_out
        self.num_embed = d_out
        self.vq_layer = VectorQuantizer(self.num_embed,
                                        self.emb_dim,
                                        beta,
                                        jitter)

        
    def forward(self, x,wont_use=None):
        x_in = x.reshape(x.shape[0],1,x.shape[1],x.shape[2])
        h = torch.relu(self.in_layer(x_in))
        h = torch.relu(self.hd_layer(h))
        h = torch.relu(self.hd_layer2(h))
        h = self.r_block(h)
        h = h.permute(0,2,3,1)
        h_shape = h.shape
        h = h.reshape(x.shape[0],x.shape[1],-1)
        p = self.mu_layer(h)
        p = p.reshape(h_shape)
        c = self.c_layer(h)

        # discrete
        vq_loss,quantized_inputs,perplexity, encodings = self.vq_layer(p)
        quantized_inputs = quantized_inputs.reshape(*h.shape[:2],-1)
        

        return quantized_inputs, vq_loss, c
    
    

class Decoder_conv(nn.Module):
    def __init__(self,l_in1,l_out,hidden_dim,d_out,mod=0):
        super(Decoder_conv, self).__init__()
        self.in_layer1 = nn.Conv2d(l_in1,l_in1//2,[1,3],stride=[1,1],padding=[0,1])# Linear(l_in,hidden_dim)
        self.in_layer2 = nn.Conv2d(l_in1,l_in1//2,[1,3],stride=[1,1],padding=[0,1])# Linear(l_in,hidden_dim)
        self.r_block = ResidualBlock(l_in1,2) #set num layer hidden to 2 for now
        self.hd_layer = nn.ConvTranspose2d(l_in1,l_in1//2,[1,4],stride=[1,2],padding=[0,1])
        self.hd_layer1 = nn.ConvTranspose2d(l_in1//2,l_out,[1,4],stride=[1,2],padding=[0,1])
        self.out_layer = nn.Linear(32, 32)
        self.d_out = d_out
        self.l_in1 = l_in1
        
    def forward(self, x,y,state=None):
        x = x.reshape(*x.shape[:2],-1,self.l_in1)
        y = y.reshape(x.shape)
        x = x.permute(0,3,1,2)
        y = y.permute(0,3,1,2)
        x = torch.relu(self.in_layer1(x))
        y = torch.relu(self.in_layer2(y))
        h = torch.cat((x, y), dim=1)
        h = self.r_block(h)
        h = torch.relu(self.hd_layer(h))
        h = torch.relu(self.hd_layer1(h))
        h = h.reshape(h.shape[0],h.shape[2],h.shape[3])
        x_reconstr = self.out_layer(h)
        return x_reconstr
    


class Decoder(nn.Module):
    def __init__(self,l_in1,l_out,hidden_dim,d_out,mod=0):
        super(Decoder, self).__init__()
        self.in_layer1 = nn.Conv2d(l_in1*4,l_in1*2,[1,3],stride=[1,1],padding=[0,1])# Linear(l_in,hidden_dim)
        self.in_layer2 = nn.Conv2d(l_in1*4,l_in1*2,[1,3],stride=[1,1],padding=[0,1])# Linear(l_in,hidden_dim)
        self.r_block = ResidualBlock(l_in1*4,2) #set num layer hidden to 2 for now
        self.hd_layer = nn.ConvTranspose2d(l_in1*4,l_in1*2,[1,4],stride=[1,2],padding=[0,1])
        self.hd_layer1 = nn.ConvTranspose2d(l_in1*2,l_out,[1,4],stride=[1,2],padding=[0,1])
        # self.in_layer1 = nn.Linear(l_in1*d_out+mod, int(hidden_dim/2))
        # self.in_layer2 = nn.Linear(l_in1*d_out, int(hidden_dim/2))
        # self.hd_layer = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(8, 8)
        self.d_out = d_out
        self.l_in1 = l_in1
        
    def forward(self, x,y,state=None):
        x = x.reshape(*x.shape[:2],-1,self.l_in1*4)
        y = y.reshape(x.shape)
        x = x.permute(0,3,1,2)
        y = y.permute(0,3,1,2)
        x = torch.relu(self.in_layer1(x))
        y = torch.relu(self.in_layer2(y))
        h = torch.cat((x, y), dim=1)
        h = self.r_block(h)
        h = torch.relu(self.hd_layer(h))
        h = torch.relu(self.hd_layer1(h))
        h = h.reshape(h.shape[0],h.shape[2],h.shape[3])
        # x_reconstr = torch.tanh(self.out_layer(h))
        x_reconstr = self.out_layer(h)
        return x_reconstr
    
