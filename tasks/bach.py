
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset, random_split
from torch.utils.data import DataLoader, Dataset

import numpy as np
from numpy.random import default_rng


MEAN, STD = torch.tensor([0.7169, 0.6170, 0.8427]), torch.tensor([0.1661, 0.1885, 0.1182]) # calculated on training and validation set


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
        R2n3f,u8}


class Bach:
    
    def __init__(self, root, train_transform, valid_transform, download=False, split_pcts=[0.6, 0.2, 0.2]):
        
        assert len(split_pcts) == 3 and np.sum(split_pcts) == 1.0
        
        self.root = root
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.split_pcts = split_pcts
        
        if download:
            self.download_data()
            self.root = self.root/'ICIAR2018_BACH_Challenge/Photos' # extend root directory to point to images
            
        self.train_ds, self.valid_ds, self.test_ds = self.prepare_datasets()
        
    
    def download_data(self):
        url = 'https://zenodo.org/record/3632035/files/ICIAR2018_BACH_Challenge.zip'
        download_and_extract_archive(url, self.root)
    
    
    def prepare_datasets(self):
        
        ds = ImageFolder(self.root)
        n_samples = len(ds)

        split_lens = [
            int(np.floor(n_samples * self.split_pcts[0])), # train
            int(np.floor(n_samples * self.split_pcts[1])), # valid
            int(np.floor(n_samples * self.split_pcts[2]))  # test
        ]

        train_ds, valid_ds, test_ds = random_split(ds, split_lens, generator=torch.Generator().manual_seed(42))
        train_ds = DatasetFromSubset(train_ds, self.train_transform)
        valid_ds = DatasetFromSubset(valid_ds, self.valid_transform)
        test_ds  = DatasetFromSubset(test_ds,  self.valid_transform)
        
        return train_ds, valid_ds, test_ds
    
    
    def get_dataloaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=0, sampler=None):
        
        train_dl = DataLoader(
            self.train_ds, batch_size=batch_size, sampler=sampler, 
            shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        
        valid_dl = DataLoader(
            self.valid_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory)
        
        return train_dl, valid_dl
    
    
    def get_dataloaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=0):
        
        train_dl = DataLoader(
            self.train_ds, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        
        valid_dl = DataLoader(
            self.valid_ds, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory)

        test_dl = DataLoader(
            self.test_ds, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory)
        
        return train_dl, valid_dl, test_dl