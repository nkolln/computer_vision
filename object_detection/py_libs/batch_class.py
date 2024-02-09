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
        df_anno = pd.read_csv(path_anno)
        df_anno['label'] = pd.Categorical(df_anno['label'])
        df_anno['label'] = df_anno['label'].cat.codes

        print(df_anno)
        df_anno['width'] = df_anno['x'] + df_anno['width']
        df_anno['height'] = df_anno['y'] + df_anno['height']
        
        self.df_gp = df_anno[['id','label']].groupby('id').agg(lambda x:x.tolist())
        
        test = df_anno.groupby('id').apply(lambda df: [list(x) for x in zip(df['x'], df['y'], df['width'], df['height'])]).reset_index(name='bbox')
        self.df_gp = pd.merge(self.df_gp,test,on='id')
        print(self.df_gp)
        
        self.batch_size = batch_size
        self.device = device
        
        #Select train set
        #Set train and test
        self.train_idx = np.random.choice(np.arange(self.len_imgs),int(self.len_imgs*0.9),replace=False)
        self.oh_train = self.df_gp.iloc[self.train_idx].reset_index(drop=True)
        self.oh_train_tensor = torch.stack([torch.load(f'{self.path_img}{img}.pt').to(self.device) for img in self.oh_train['id'].tolist()])

        self.oh_test = self.df_gp.iloc[~self.df_gp.index.isin(pd.Series(self.train_idx))].reset_index(drop=True)
        self.oh_test_tensor = torch.stack([torch.load(f'{self.path_img}{img}.pt').to(self.device) for img in self.oh_test['id'].tolist()])
        
        self.train_len = self.oh_train.shape
        self.test_len = self.oh_test.shape

        self.device = device

        print('Done Preprocessing Batch')

 

    def get_batch_train(self,transforms):
        #Get the train set
        idx = np.random.choice(np.arange(self.train_len[0]),self.batch_size,replace=False)
        #Select the batches
        x = self.oh_train_tensor[idx]
        #select the one hot encodings
        one_hot = self.oh_train.iloc[idx]
        targets = []
        for i in range(x.shape[0]):
            d = {}
            d['boxes'] = torch.tensor(one_hot['bbox'].iloc[i],device=self.device)
            d['labels'] = torch.tensor(one_hot['label'].iloc[i],device=self.device)
            targets.append(d)

        # print(tensor_to_image(df,self.path_img))
        
        #Apply the transformations
        x = transforms(x)

        # print(tensor_to_image(t_img,None))

        #BxCxWxH and #BxW
        return(x,targets)

    def get_batch_test(self,transforms):
        #Get the train set
        idx = np.random.choice(np.arange(self.test_len[0]),self.batch_size,replace=False)
        #Select the batches
        x = self.oh_test_tensor[idx]
        #select the one hot encodings
        one_hot = self.oh_test.iloc[idx]

        # print(tensor_to_image(df,self.path_img))
        
        #Apply the transformations
        x = transforms(x)
        #BxCxWxH and #BxW
        return(x,one_hot)
