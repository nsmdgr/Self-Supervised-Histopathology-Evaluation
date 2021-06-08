
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import numpy as np
from numpy.random import default_rng

MEAN, STD = torch.tensor([0.7169, 0.6170, 0.8427]), torch.tensor([0.1661, 0.1885, 0.1182]) # calculated on training and validation set

class Bach:
    
    def __init__(self, root, train_transform, valid_transform, download=False, valid_percent=0.2, shuffle=True):
        
        self.root = root
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.valid_percent = valid_percent
        self.shuffle = shuffle
        
        if download:
            self.download_data()
            self.root = self.root/'ICIAR2018_BACH_Challenge/Photos' # extend root directory to point to images
            
        self.train_ds, self.valid_ds, self.train_sampler, self.valid_sampler = self.prepare_datasets()
        
    
    def download_data(self):
        url = 'https://zenodo.org/record/3632035/files/ICIAR2018_BACH_Challenge.zip'
        download_and_extract_archive(url, self.root)
    
    
    def prepare_datasets(self):
        train_ds = ImageFolder(self.root, self.train_transform)
        valid_ds = ImageFolder(self.root, self.valid_transform)
        
        num_train = len(train_ds)
        indices   = list(range(num_train))
        split     = int(np.floor(self.valid_percent * num_train))
        
        if self.shuffle:
            rng = default_rng(seed=101)
            rng.shuffle(indices)
        
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        return train_ds, valid_ds, train_sampler, valid_sampler
    
    
    def get_dataloaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=0):
        train_dl = DataLoader(
            self.train_ds, batch_size=batch_size, sampler=self.train_sampler, 
            shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        
        valid_dl = DataLoader(
            self.valid_ds, batch_size=batch_size, sampler=self.valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory)
        
        return train_dl, valid_dl