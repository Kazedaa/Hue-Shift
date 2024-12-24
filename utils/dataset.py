import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
from PIL import Image, ImageCms
import numpy as np

class Dataset(Dataset):
    def __init__(self ,root ,transform=None ,lab=False):
        self.root = root
        self.files = os.listdir(root)
        self.len = len(self.files)
        if transform is not None:
            self.transforms = transforms.Compose(transform)
        else:
            self.transforms = None
        
        self.lab = lab

        srgb_p = ImageCms.createProfile("sRGB")
        lab_p  = ImageCms.createProfile("LAB")
        self.rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")

        
    def __getitem__(self , i):
        file = self.files[i]
        image = Image.open(f'{self.root}/{file}').convert('RGB')

        if self.lab:
            image = ImageCms.applyTransform(image, self.rgb2lab)

        if self.transforms is not None:
            image = self.transforms(image)

        return image

    def __len__(self):
        return self.len
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    transform = [
        transforms.ToTensor(),
        transforms.Resize((64,64), Image.BICUBIC),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]


    data_loader = DataLoader(
        Dataset("data",transform, lab=True),
        batch_size = 16,
        shuffle = True,
        num_workers = 2
    )
    
    
    im = next(iter(data_loader))

    L,A,B = torch.split(im,[1,1,1],dim=1)

    print(L.shape)