import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import numpy.fft as nf
import os
import h5py
import random

class TrainingDataset(Dataset):
    def __init__(self, dataPath, maskPath):
        self.dataPath = dataPath
        self.maskPath = maskPath
        self.img = []
        self.mask = []
        
        for path, _, fns in os.walk(self.dataPath):
            for fn in fns:
                im = h5py.File(os.path.join(path, fn))['im_ori'][()]
                img = Image.fromarray(im)
                self.img.append(img)
                
        for path, _, fns in os.walk(self.maskPath):
            for fn in fns:
                mask = h5py.File(os.path.join(path, fn))['mask'][()]
                self.mask.append(nf.ifftshift(mask))

    def collate_fn(self, batch):
        train = []
        label = []
        for img in batch:
            kspace_full = nf.fft2(img.numpy())
            idx = random.choice(range(4))
            train_data = self.mask[idx]*kspace_full
            real = torch.from_numpy(train_data.real).view(1,256,256,1)
            imag = torch.from_numpy(train_data.imag).view(1,256,256,1)
            train_data = torch.cat((real, imag), 3).float()
            label.append(img.unsqueeze(0))
            train.append(train_data.unsqueeze(0))
        return torch.cat(train, 0), torch.cat(label, 0)
        
    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
        if random.random() > 0.5:
            img = transforms.functional.rotate(img, 90)
        if random.random() > 0.5:
            img = transforms.functional.vflip(img)
        img = transforms.functional.to_tensor(img)
        return img

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    dataPath = './data/train_ori'
    maskPath = './data/mask'
    dataset = TrainingDataset(dataPath, maskPath)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=48, shuffle=True, collate_fn=dataset.collate_fn)

    for epoch in range(1):
        for batch_n, (train, label) in enumerate(dataloader):
            print(train.size())
            print(label.size())
            break