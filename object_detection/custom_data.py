from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import cv2

class CustomDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(f'data/train2017/{self.paths[index]}')
        image = cv2.resize(image, (480,640),interpolation = cv2.INTER_LINEAR)
        # Normalize the image
        image = image / 255.0
        return image