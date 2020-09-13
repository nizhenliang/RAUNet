import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image as Image

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
y_trans=transforms.ToTensor()

class Load_Dataset(Dataset):
    def __init__(self, filenames):
        self.file_names = filenames

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        down_sample = 2
        img_file_name = self.file_names[idx]
        ori_image = load_image(img_file_name)
        image = x_transforms(ori_image)
        image = F.max_pool2d(image, kernel_size=(down_sample, down_sample), stride=down_sample, padding=0)
        image = F.pad(image, (0, 0, 2, 2), 'constant', 0)

        mask = load_mask(img_file_name)
        mask = mask[np.newaxis, :, :]
        labels = torch.from_numpy(mask).float()
        labels = F.max_pool2d(labels, kernel_size=(down_sample, down_sample), stride=down_sample, padding=0)
        labels = F.pad(labels, (0, 0, 2, 2), 'constant', 0)
        labels = labels.squeeze()
        return image, labels


def load_image(path):
    img_x = Image.open(path)
    return img_x



def load_mask(path):
    new_path=path.replace('image', 'label')
    mask = cv2.imread(new_path, 0)
    mask=mask//20

    return mask.astype(np.uint8)
