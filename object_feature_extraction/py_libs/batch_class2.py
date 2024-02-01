import numpy as np
import random
import torch
import pickle
import time
# from torchvision import transforms
import torchvision
import os
import pandas as pd
from torchvision.transforms import v2
import PIL
import matplotlib.pyplot as plt
import time

def tensor_to_image(tensor,path):
    print(tensor.shape)
    if path is not None:
        tensor = torch.load(f'{path}{tensor["filename"].values[0]}.pt')
    else:
        tensor = tensor[0].detach().cpu()
    # print(tensor.shape)
    image = tensor.permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(image)
    plt.show()


class batch_main():
    def __init__(self,path_img,path_anno,batch_size,device) -> None:
        #check if inputs are the same
        self.path_img = path_img
        self.dirs_img = os.listdir(path_img)
        self.len_imgs = len(self.dirs_img)

        
        # self.dirs_anno = os.listdir(path_anno)
        # self.len_anno = len(self.dirs_anno)

        #Load annotations and generate one hot encoding of the input
        df_anno = pd.read_csv(path_anno)[['filename','region_attributes']]
        df_anno['filename'] = df_anno['filename'].apply(lambda x:x[:-4]) #nice not to have to deal with later
        df_gp = df_anno.groupby('filename').aggregate(lambda x:x.unique().tolist())
        self.one_hot = pd.get_dummies(df_gp['region_attributes'].explode()).groupby('filename').sum().reset_index()
        print(self.one_hot)

        
        self.batch_size = batch_size
        self.device = device
        
        #Select train set
        #Set train and test
        self.train_idx = np.random.choice(np.arange(self.len_imgs),int(self.len_imgs*0.9),replace=False)
        self.oh_train = self.one_hot.iloc[self.train_idx].reset_index(drop=True)
        self.oh_train_tensor = torch.stack([torch.load(f'{self.path_img}{img}.pt').to(self.device) for img in self.oh_train['filename'].tolist()])
        self.oh_train_anno = torch.tensor(self.oh_train.loc[:, self.oh_train.columns != 'filename'].values,device=device)

        self.oh_test = self.one_hot.iloc[~self.one_hot.index.isin(pd.Series(self.train_idx))].reset_index(drop=True)
        self.oh_test_tensor = torch.stack([torch.load(f'{self.path_img}{img}.pt').to(self.device) for img in self.oh_test['filename'].tolist()])
        self.oh_test_anno = torch.tensor(self.oh_test.loc[:, self.oh_test.columns != 'filename'].values,device=device)
        
        self.train_len = self.oh_train.shape
        self.test_len = self.oh_test.shape

        print('Done Preprocessing Batch')

 

    def get_batch_train(self,transforms):
        #Get the train set
        idx = np.random.choice(np.arange(self.train_len[0]),self.batch_size,replace=False)
        #Select the batches
        x = self.oh_train_tensor[idx]
        #select the one hot encodings
        one_hot = self.oh_train_anno[idx]

        # print(tensor_to_image(df,self.path_img))
        
        #Apply the transformations
        x = transforms(x)

        # print(tensor_to_image(t_img,None))

        #BxCxWxH and #BxW
        return(x,one_hot)

    def get_batch_test(self,transforms):
        #Get the train set
        idx = np.random.choice(np.arange(self.test_len[0]),self.batch_size,replace=False)
        #Select the batches
        x = self.oh_test_tensor[idx]
        #select the one hot encodings
        one_hot = self.oh_test_anno[idx]

        # print(tensor_to_image(df,self.path_img))
        
        #Apply the transformations
        x = transforms(x)
        #BxCxWxH and #BxW
        return(x,one_hot)
