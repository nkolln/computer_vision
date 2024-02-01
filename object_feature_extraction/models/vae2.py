import torch.nn as nn
import torch

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
        e_latent_loss = torch.nn.functional.mse_loss(quantized.detach(), latents)
        q_latent_loss = torch.nn.functional.mse_loss(quantized, latents.detach())
        loss = q_latent_loss + self.beta * e_latent_loss
        
        quantized = latents + (quantized - latents).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


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
        self.in_layer = nn.Conv2d(in_size,out_size//2,4,stride=2,padding=1)
        self.hd_layer = nn.Conv2d(out_size//2,out_size,7,stride=4,padding=2)
        self.r_block = ResidualBlock(out_size,2)
        self.hd_layer2 = nn.Conv2d(out_size,out_size,7,stride=4,padding=2)
        self.hd_layer3 = nn.Conv2d(out_size,out_size,3,stride=2,padding=1)

        self.vq_layer = VectorQuantizer(d_out,
                                        out_size,
                                        beta,
                                        jitter)
        

        
    def forward(self, x):
        h = torch.relu(self.in_layer(x))
        h = torch.relu(self.hd_layer(h))
        h = torch.relu(self.hd_layer2(h))
        h = torch.relu(self.hd_layer3(h))
        h = self.r_block(h)

        # discrete
        vq_loss,quantized_inputs,_, _ = self.vq_layer(h)
        
        # #continuous
        # vq_loss = kl_loss(p,c)
        # quantized_inputs = p

        return quantized_inputs, vq_loss,_

class Decoder_conv(nn.Module):
    def __init__(self,in_size,out_size,hidden_dim=None,mod=0):
        super(Decoder_conv, self).__init__()
        self.in_layer1 = nn.Conv2d(in_size,in_size,3,stride=1,padding=1)# Linear(l_in,hidden_dim)
        self.r_block = ResidualBlock(in_size,2) #set num layer hidden to 2 for now
        self.hd_layer = nn.ConvTranspose2d(in_size,in_size,4,stride=2,padding=1)
        self.hd_layer1 = nn.ConvTranspose2d(in_size,in_size,6,stride=4,padding=1)
        self.hd_layer2 = nn.ConvTranspose2d(in_size,in_size//2,6,stride=4,padding=1)
        self.hd_layer3 = nn.ConvTranspose2d(in_size//2,out_size,4,stride=2,padding=1)
        
    def forward(self, x):
        h = torch.relu(self.in_layer1(x))
        h = self.r_block(h)
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
        self.hd_layer2 = nn.Linear(hidden_dim,32*54*72)
        self.out_size = out_size

        self.vq_layer = VectorQuantizer(d_out,
                                        out_size,
                                        beta,
                                        jitter)
        

        
    def forward(self, x):
        x = x.float()
        h = torch.relu(self.in_layer(x))
        h = torch.relu(self.hd_layer(h))
        h = torch.relu(self.hd_layer2(h))
        h = h.reshape(h.shape[0],self.out_size,54,72)
        # discrete
        vq_loss,quantized_inputs,_, _ = self.vq_layer(h)
        
        # #continuous
        # vq_loss = kl_loss(p,c)
        # quantized_inputs = p

        return quantized_inputs, vq_loss,_

class Decoder(nn.Module):
    def __init__(self,in_size,out_size,d_out=256,hidden_dim=512):
        super(Decoder, self).__init__()
        self.in_layer = nn.Linear(32*54*72,hidden_dim)
        self.hd_layer = nn.Linear(hidden_dim,hidden_dim//2)
        self.hd_layer2 = nn.Linear(hidden_dim//2,in_size)
        self.out_layer = nn.Linear(in_size,in_size)
        
    def forward(self, x):
        x = x.reshape(*x.shape[:1],-1)
        h = torch.relu(self.in_layer(x))
        h = torch.relu(self.hd_layer(h))
        h = torch.relu(self.hd_layer2(h))
        x_reconstr = torch.sigmoid(self.out_layer(h))
        # x_reconstr = torch.where(x_reconstr>0.5,1,0)
        return x_reconstr