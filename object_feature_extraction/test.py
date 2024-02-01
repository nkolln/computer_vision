import numpy as np
import random
import torch
import pickle
import time
# from torchvision import transforms
import torchvision
import os
import pandas as pd


class batch_main():
    def __init__(self,path_img,path_anno,batch_size,device) -> None:
        #check if inputs are the same
        self.dirs_img = os.listdir(path_img)
        self.len_imgs = len(self.dirs_img)

        
        # self.dirs_anno = os.listdir(path_anno)
        # self.len_anno = len(self.dirs_anno)

        #Load annotations and generate one hot encoding of the input
        df_anno = pd.read_csv(path_anno)[['filename','region_attributes']]
        df_anno['filename'] = df_anno['filename'].apply(lambda x:x[:-4]) #nice not to have to deal with later
        df_gp = df_anno.groupby('filename').aggregate(lambda x:x.unique().tolist())
        self.one_hot = pd.get_dummies(df_gp['region_attributes'].explode()).groupby('filename').sum().reset_index()
        
        self.batch_size = batch_size
        self.device = device
        
        #Select train set
        self.train_idx = np.random.choice(np.arange(self.len_imgs),int(self.len_imgs*0.9),replace=False)
        self.oh_train = self.one_hot.iloc[self.train_idx].reset_index(drop=True)
        self.oh_test = self.one_hot.iloc[~self.one_hot.index.isin(pd.Series(self.train_idx))].reset_index(drop=True)
        
        self.train_len = self.oh_train.shape
        self.test_len = self.oh_test.shape

        print(self.oh_train)
        print(self.oh_test)
 

    def get_batch_train(self):
        print(self.dirs_img)
        #Get the train set
        idx = np.random.choice(np.arange(self.train_len[0]),self.batch_size,replace=False)
        df = self.oh_train.iloc[idx]
        print(df)
        t_img = torch.empty(1,3, 3456, 4608)
        [tor(torch.load())]
        print(torch.stack(df['filename']))
        #Apply the transformations

        #BxCxWxH and #BxW
        return(img,one_hot)
        pass

    def get_batch_test(self):
        #Get the test set
        pass
