#G9 is using the new actuator strength data

import numpy as np
from py_libs.utils import str2bool
from py_libs.batch_class import batch_main

import torch
import time
import argparse 
import os

from torch.utils.data import Dataset, DataLoader
from models.base_cnn import CNN_class
from models.pretrained import CNN_pretrained

from custom_data import CustomDataset


def vae(
        variant,
        FLAGS,
    ):
    device = variant.get('device', 'cuda')

    path_img = '/home/nkolln/computer_vision/object_detection/data/images/'
    path_anno = '/home/nkolln/computer_vision/object_detection/data/anno.csv'
    dir_dirt = 'data/train2017'

    #create batch class
    lst_image = [file for file in sorted(os.listdir(dir_dirt)) if 'jpg' in file]
    dataset = CustomDataset(paths=lst_image)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    batch = batch_main(path_img,path_anno,variant['batch_size'],device)

    #create transformation
    

    criterion1 = torch.nn.MSELoss()
    criterion2 = torch.nn.BCELoss()

    #load model and its hyperparameters
    cnn = CNN_pretrained().to(device)
    
    opt_param = list(cnn.parameters())
    optimizer  = torch.optim.Adam(opt_param, lr=0.0001,betas = (0.9,0.999))
    torch.autograd.set_detect_anomaly(True)   
    time1 = time.time()                          
    #Load eval set
    for i in range(variant['max_iters']):
        cnn.train()
        vq_loss_sum = 0
        for j in range(variant['num_steps_per_iter']):
            
            optimizer.zero_grad()
            img,anno = batch.get_batch_train(transforms_train)
            pred = cnn(img,anno)
            loss = pred['loss_classifier'] + pred['loss_rpn_box_reg']

            loss.backward()
            
            optimizer.step()
            vq_loss_sum+=loss.item()
        print(f'Iter: {i}  \tTime_total: {round((time.time()-time1),2)}  \tLoss: {vq_loss_sum/(j+1)}')
        
        cnn.train()
        with torch.no_grad():
            loss = 0
            for _ in range(variant['num_eval_episodes']):
                img,anno = batch.get_batch_train(transforms_test)
                pred = cnn(img,anno)
                loss = pred['loss_classifier'] + pred['loss_rpn_box_reg']
        print((f'Eval Loss: {loss/variant["num_eval_episodes"]}\n'))

    torch.save(cnn,'cnn_to_param.pt')


    #load trainer with eval

    #Get results and test final result

    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=50)
    parser.add_argument('--num_steps_per_iter', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    FLAGS = args

    vae(variant=vars(args), FLAGS = FLAGS)