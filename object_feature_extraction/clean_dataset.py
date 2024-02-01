
import os
from torchvision.io import read_image
import pandas as pd
import torch
from torchvision.transforms import v2

transforms = v2.Compose([
                # v2.RandomResizedCrop(size=(3456, 4608), antialias=True),
                v2.Resize(size=(3456//16, 4608//16), antialias=True), #216 288
                # v2.ToDtype(torch.float32, scale=True),
            ])
if __name__ == '__main__':
    path_img = '/home/nkolln/orbisk_test/dataset/'
    
    dirs_img = os.listdir(path_img)

    for img in dirs_img:
        img_load = read_image(f'{path_img}{img}')
        print(img_load.shape)
        if img_load.shape[-1]!= 4608:
            print('failed')
            img_load = img_load.permute(0,2,1)
        img_load = transforms(img_load)
        print(img_load.shape)
        torch.save(img_load,f'img_clean/{img[:-4]}.pt')