import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_and_extract_archive, download_url
import h5py


MEAN, STD = torch.tensor([0.7008, 0.5384, 0.6916]), torch.tensor([0.2350, 0.2774, 0.2129]) # calculated on training set

class PatchCamelyonDataset(Dataset):

    def __init__(self, root, transform, mode='train'):
        super().__init__()

        assert mode in ['train', 'valid', 'test']
        
        self.root = root
        self.transform = transform
        self.mode = mode

        self.X = h5py.File(root/f'camelyonpatch_level_2_split_{mode}_x.h5', 'r').get('x')
        self.y = h5py.File(root/f'camelyonpatch_level_2_split_{mode}_y.h5', 'r').get('y')

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        x, y = self.transform(x), y.item()
        return x, y

    def __len__(self):
        return len(self.X)


class PatchCamelyon:
    
    def __init__(self, root, train_transform, valid_transform, download=False):
        
        self.root = root
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        
        if download:
            self.download_data()
            
        self.train_ds, self.valid_ds, self.test_ds = self.prepare_datasets()
        
    def download_data(self):
        base_url = 'https://zenodo.org/record/2546921/files/'
        for mode in ['train', 'valid', 'test']:
            download_url(base_url + f'camelyonpatch_level_2_split_{mode}_meta.csv', self.root)
            for xy in ['x','y']: 
                download_and_extract_archive(base_url + f'camelyonpatch_level_2_split_{mode}_{xy}.h5.gz', self.root)
    
    def prepare_datasets(self):
        train_ds = PatchCamelyonDataset(self.root, transform=self.train_transform, mode='train')
        valid_ds = PatchCamelyonDataset(self.root, transform=self.valid_transform, mode='valid')
        test_ds  = PatchCamelyonDataset(self.root, transform=self.valid_transform, mode='test')
        
        return train_ds, valid_ds, test_ds
    
    def get_dataloaders(self, batch_size, shuffle=True, pin_memory=True, num_workers=0):
        
        train_dl = DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory)
        
        valid_dl = DataLoader(
            self.valid_ds, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory)
        
        test_dl = DataLoader(
            self.test_ds, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory)
        
        return train_dl, valid_dl, test_dl

