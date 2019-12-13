import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import tools
import random

class TrainingDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.imageSize = (180, 180)
        self.images = []

        # blur kernel motion with length 21 and angle 45 deg  
        # pad it into imagesize and shift it for further use.
        k = tools.fspecial('motion', 21, 45)
        kshift = tools.pad_and_shift(k, self.imageSize)
        kshift = torch.from_numpy(kshift).float().unsqueeze(0).unsqueeze(0)
        self.K = torch.rfft(kshift, 2, onesided=False)

        for path, _, fns in os.walk(self.path):
            for fn in fns:
                img = Image.open(os.path.join(self.path, fn))
                self.images.append(img.copy())
                img.close()
    
    def collate_fn(self, batch):
        originImg = []
        blurredNoisyImg = []
        for img in batch:
            originImg.append(img.unsqueeze(0))
            noise = torch.randn(img.size()) * (5.7020/255)
            # generate blurry image directly using multiplication in frequenvy domain.
            blurred = torch.irfft(tools.complex_multiplication(torch.rfft(img.unsqueeze(0), 2, onesided=False), self.K), 2, onesided=False)
            blurredNoisyImg.append(blurred + noise.unsqueeze(0))
        return torch.cat(originImg, 0), torch.cat(blurredNoisyImg, 0)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=self.imageSize)
        img = transforms.functional.crop(img, i, j, h, w)
        # Random horizontal flipping
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
        # Random rotation by 90 degree
        if random.random() > 0.5:
            img = transforms.functional.rotate(img, 90)
        # Random vertical flipping
        if random.random() > 0.5:
            img = transforms.functional.vflip(img)
        img = transforms.functional.to_tensor(img)
        return img

class TestDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.imageSize = (180, 180)
        self.images = []

        # blur kernel motion with length 21 and angle 45 deg  
        # pad it into imagesize and shift it for further use.
        k = tools.fspecial('motion', 21, 45)
        kshift = tools.pad_and_shift(k, self.imageSize)
        kshift = torch.from_numpy(kshift).float().unsqueeze(0).unsqueeze(0)
        self.K = torch.rfft(kshift, 2, onesided=False)

        for path, _, fns in os.walk(self.path):
            for fn in fns:
                img = Image.open(os.path.join(self.path, fn))
                self.images.append(img.copy())
                img.close()
    
    def collate_fn(self, batch):
        originImg = []
        blurredNoisyImg = []
        for img in batch:
            originImg.append(img.unsqueeze(0))
            noise = torch.randn(img.size()) * (5.7020/255)
            # generate blurry image directly using multiplication in frequenvy domain.
            blurred = torch.irfft(tools.complex_multiplication(torch.rfft(img.unsqueeze(0), 2, onesided=False), self.K), 2, onesided=False)
            blurredNoisyImg.append(blurred + noise.unsqueeze(0))
        return torch.cat(originImg, 0), torch.cat(blurredNoisyImg, 0)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=self.imageSize)
        img = transforms.functional.crop(img, i, j, h, w)
        img = transforms.functional.to_tensor(img)
        return img

    
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    path = "../dataset/BSDS500/deblur_motion/train"
    dataset = TrainingDataset(path)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=48, shuffle=True, collate_fn=dataset.collate_fn)

    for epoch in range(1):
        for batch_n, (img, noisyImg) in enumerate(dataloader):
            print(img.size())
            print(noisyImg.size())
            break
