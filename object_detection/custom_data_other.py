from torch.utils.data import Dataset
import cv2
from torchvision.io import read_image


    
class CustomImages(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = read_image(f'data/train2017/{self.paths[index]}')
        # image = cv2.imread(f'data/train2017/{self.paths[index]}')
        return image



class CustomDataset(Dataset):
    def __init__(self, paths_img,anno,lst_cat,transforms):
        self.paths_img = paths_img
        self.anno = anno
        self.cat = lst_cat
        self.transforms = transforms

    def __len__(self):
        return len(self.paths_img)

    def __getitem__(self, index):
        image = read_image(f'data/train2017/{self.paths_img[index]}')
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = self.transforms(image)
        return image,self.anno[index],self.cat[index]
    


# from torch.utils.data import Dataset
# import cv2
# from torchvision.io import read_image


# class CustomImages(Dataset):
#     def __init__(self, paths, transform=None):
#         self.paths = paths

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, index):
#         image = cv2.imread(f'data/train2017/{self.paths[index]}')
#         image = cv2.resize(image, (480,640),interpolation = cv2.INTER_LINEAR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # Normalize the image
#         image = image / 255.0
#         return image
    


# class CustomDataset(Dataset):
#     def __init__(self, paths_img,anno,lst_cat):
#         self.paths_img = paths_img
#         self.anno = anno
#         self.cat = lst_cat

#     def __len__(self):
#         return len(self.paths_img)

#     def __getitem__(self, index):
#         image = cv2.imread(f'data/train2017/{self.paths_img[index]}')
#         image = cv2.resize(image, (480,640),interpolation = cv2.INTER_LINEAR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # Normalize the image
#         image = image / 255.0
#         # print(image.shape)
#         # image = Image.fromarray(image)
#         return image,self.anno[index],self.cat[index]