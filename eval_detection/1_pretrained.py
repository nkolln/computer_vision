#G9 is using the new actuator strength data

import numpy as np
from py_libs.utils import str2bool
from custom_data import CustomImages



import torch
import time
import argparse 
import cv2
import os
import random

from models.pretrained import CNN_pretrained
from torch.utils.data import DataLoader

def vae(
        variant,
        FLAGS,
    ):
    device = variant.get('device', 'cuda')
    dir_dirt = 'data/train2017'

    lst_image = [file for file in sorted(os.listdir(dir_dirt)) if 'jpg' in file]

    #load model and its hyperparameters
    cnn = CNN_pretrained(variant).to(device)
    

    # cnn = torch.load('cnn_fine_tuned.pt')

    cnn.eval()

    #Load data
    cData = CustomImages(lst_image)
    data_loader = DataLoader(cData, batch_size=1, shuffle=False, num_workers=4)
    for i,images in enumerate(data_loader):
        
        
        if i > 100:
            break
        images = images.to(device).float().permute(0,3,1,2)

        #Augmentations

        predictions  = cnn(images)
        cv2_image = images.reshape(*images.shape[1:]).permute(1,2,0).to('cpu').numpy()
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        idx = torch.where(predictions[0]['scores']>0.5)
        preds = predictions[0]
        for (x,y,x2,y2),cls in zip(preds['boxes'][idx].cpu().detach().numpy(),preds['labels'][idx].cpu().detach().numpy()):
            x=int(x);y=int(y);x2=int(x2);y2=int(y2)
            cv2_image = cv2.rectangle(cv2_image, (x, y), (x2, y2), (36,255,12), 1)
            cv2.putText(cv2_image, str(cls), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 1)

        if not os.path.isdir(f'data/{variant["out_file"]}'):    
            os.makedirs(f'data/{variant["out_file"]}', exist_ok=True)
        cv2.imwrite(f'data/{variant["out_file"]}/{i}.jpg', cv2_image*255) 

        # Display the image using cv2
        # cv2.imshow("Image", cv2_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--out_file', type=str, default='q1_images')
    parser.add_argument('--new_load', type=str2bool, default='False')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    FLAGS = args

    vae(variant=vars(args), FLAGS = FLAGS)