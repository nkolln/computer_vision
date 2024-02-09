
    


from torch.utils.data import Dataset
import cv2
from torchvision.io import read_image
import random
import numpy as np


class CustomImages(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(f'data/train2017/{self.paths[index]}')
        image = cv2.resize(image, (640,480))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #Removes some blur
        image = cv2.GaussianBlur(image, (5, 5),0)

        # Normalize the image
        image = image / 255.0
        
        return image
    


class CustomDataset(Dataset):
    def __init__(self, paths_img,anno,lst_cat):
        self.paths_img = paths_img
        self.anno = anno
        self.cat = lst_cat

    def __len__(self):
        return len(self.paths_img)

    def __getitem__(self, index):
        image = cv2.imread(f'data/train2017/{self.paths_img[index]}')
        image = cv2.resize(image, (640,480))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #Removes some blur
        image = cv2.GaussianBlur(image, (5, 5),0)

        # Normalize the image
        image = image / 255.0
        return image,self.anno[index],self.cat[index]