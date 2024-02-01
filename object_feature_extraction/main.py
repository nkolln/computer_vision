#G9 is using the new actuator strength data

from torchvision.transforms import v2

import numpy as np
from utils import str2bool,kl_loss
from batch_class2 import batch_main

import torch
import time
from PIL import Image

import argparse 

from models.vae3 import Encoder_conv,Encoder,Decoder_conv,Decoder

def vae(
        variant,
        FLAGS,
    ):
    device = variant.get('device', 'cuda')

    path_img = '/home/nkolln/orbisk_test/img_clean/'
    path_anno = '/home/nkolln/orbisk_test/annotations/annotations.csv'

    # df_img = pd.read_csv('/home/nkolln/orbisk_test/dataset/IMG_20190801_130547.jpg')
    # img = Image.open('/home/nkolln/orbisk_test/dataset/IMG_20190801_130547.jpg')

    #create batch class
    batch = batch_main(path_img,path_anno,variant['batch_size'],device)

    criterion = torch.nn.BCELoss()

    
    transforms_train = v2.Compose([
        # v2.Resize(size=(3456//16, 4608//16), antialias=True), #216 288
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
    ])
    transforms_test = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
    ])

    #load model and its hyperparameters
    img_encoder = Encoder_conv(3,32).to(device)
    img_decoder = Decoder_conv(32,3).to(device)
    anno_encoder = Encoder(10,32).to(device)
    anno_decoder = Decoder(10,32).to(device)
    
    opt_param = list(img_encoder.parameters()) + list(img_decoder.parameters()) + list(anno_encoder.parameters()) + list(anno_decoder.parameters())
    auto_encoder_optimizer  = torch.optim.Adam(opt_param, lr=0.0001,betas = (0.9,0.999))
    torch.autograd.set_detect_anomaly(True)                             
    #Load eval set
    for i in range(variant['max_iters']):
        img_encoder.train();img_decoder.train();anno_encoder.train();anno_decoder.train()
        vq_loss_sum = 0;img_loss_sum = 0;anno_loss_sum = 0;z_loss_sum = 0;kl_loss_sum=0
        for j in range(variant['num_steps_per_iter']):
            
            auto_encoder_optimizer.zero_grad()
            img,anno = batch.get_batch_train(transforms_train)
            z_img,mu1,sp1 = img_encoder(img)
            z_anno,mu2,sp2 = anno_encoder(anno)

            kl_loss2 = kl_loss(mu1,sp1) + kl_loss(mu2,sp2)
            kl_loss_sum+=kl_loss2
            kl_loss2.backward(retain_graph=True)

            # vq_loss = vq_loss1 + vq_loss2
            da_loss =  torch.mean(torch.abs(mu1 - mu2)) + torch.mean(torch.abs(sp1 - sp2))
            da_loss.backward(retain_graph=True)

            img_zi = img_decoder(z_img)
            img_za = img_decoder(z_anno)

            anno_zi = anno_decoder(z_img)
            anno_za = anno_decoder(z_anno)
            

            z_loss = torch.mean(torch.abs(z_anno - z_img))
            z_loss.backward(retain_graph=True)
            # print(img_zi.shape,img_za.shape,anno.shape)
            # print(anno_zi.shape,anno_za.shape,img.shape)
            img_loss = (torch.mean(torch.abs(img_zi - img)) + torch.mean(torch.abs(img_za - img)))/2
            anno_loss =( criterion(anno_zi.float(),anno.float()) + criterion(anno_za.float(),anno.float()))/2
            img_loss.backward(retain_graph=True)
            anno_loss.backward(retain_graph=True)
            
            auto_encoder_optimizer.step()
            vq_loss_sum+=da_loss.item();img_loss_sum+=img_loss.item();anno_loss_sum+=anno_loss.item();z_loss_sum+=z_loss.item()
        print(f'Iter: {i}\tDA Loss {vq_loss_sum/(j+1)}\tImg Loss {img_loss_sum/(j+1)}\tAnno Loss {anno_loss_sum/(j+1)}\tZ Loss {z_loss/(j+1)}')
        
        img_encoder.eval();img_decoder.eval();anno_encoder.eval();anno_decoder.eval()
        with torch.no_grad():
            pass

    #load trainer with eval

    #Get results and test final result

    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=20)
    parser.add_argument('--num_steps_per_iter', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    FLAGS = args

    vae(variant=vars(args), FLAGS = FLAGS)