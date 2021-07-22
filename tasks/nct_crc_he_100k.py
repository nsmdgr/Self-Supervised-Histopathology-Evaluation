import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import DataLoader, random_split
from utils import DatasetFromSubset
import numpy as np

# Mean and Std calculated on training sets (found same for both normalised and non-normalised)
MEAN, STD = torch.tensor([0.7357, 0.5804, 0.7012]), torch.tensor([0.2262, 0.2860, 0.2300])

class NctCrcHe100K:
    
    def __init__(self, root, train_transform, valid_transform, valid_pct=0.2, seed=42, download=False, color_norm=True):
        
        self.root = root
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.valid_pct = valid_pct
        self.color_norm = color_norm
        self.seed = 42
        
        if download:
            self.download_data()
            
        self.train_ds, self.valid_ds = self.prepare_datasets()
        
    
    def download_data(self):
        base_url = 'https://zenodo.org/record/1214456/files/'
        download_and_extract_archive(base_url + 'NCT-CRC-HE-100K.zip', self.root)
        download_and_extract_archive(base_url + 'NCT-CRC-HE-100K-NONORM.zip', self.root)
        download_and_extract_archive(base_url + 'CRC-VAL-HE-7K.zip', self.root)
    
    
    def prepare_datasets(self):
        
        train_dir = 'NCT-CRC-HE-100K' if self.color_norm else 'NCT-CRC-HE-100K-NONORM'
        
        # train valid split
        ds = ImageFolder(self.root/train_dir)
        
        n_samples = len(ds)
        valid_len = int(np.floor(n_samples * self.valid_pct))
        train_len = n_samples - valid_len
        rnd_gen = torch.Generator().manual_seed(self.seed)
        train_ds, valid_ds = random_split(ds, [train_len, valid_len], generator=rnd_gen)
        
        # add transforms
        train_ds = DatasetFromSubset(train_ds, self.train_transform)
        valid_ds = DatasetFromSubset(valid_ds, self.valid_transform)

        # test dataset
        test_ds = ImageFolder(self.root/'CRC-VAL-HE-7K', self.valid_transform)
        
        return train_ds, valid_ds, test_ds
    
    
    def get_dataloaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=0):
        train_dl = DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        valid_dl = DataLoader(
            self.valid_ds, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory
        )

        test_dl = DataLoader(
            self.test_ds, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        return train_dl, valid_dl, test_dl