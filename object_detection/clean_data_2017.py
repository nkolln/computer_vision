import fiftyone.zoo as foz
from torchvision.transforms import v2
from torchvision.io import read_image
import torch
import os
import csv
import cv2
import numpy as np

dir_dirt = 'data/train2017'
dir_dirt2 = 'data/clean'

lst_anno = [file for file in sorted(os.listdir(dir_dirt)) if 'json' in file]
lst_image = [file for file in sorted(os.listdir(dir_dirt)) if 'jpg' in file]

transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])


for img in lst_image:
    img_ld = cv2.imread(f'{dir_dirt}/{img}')
    img_ld = cv2.resize(img_ld, (480,640),interpolation = cv2.INTER_LINEAR)
    # flip = cv2.flip(torch.tensor(img_ld), 1)
    img_ld = img_ld/255
    
    img_ld = cv2.imwrite(f'{dir_dirt2}/{img}')
    # img_ld = np.transpose(img_ld,np.argsort(img_ld.shape))
    # lst.append(torch.tensor(img_ld))
    # flip = cv2.flip(img_ld, 1)