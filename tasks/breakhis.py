import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import default_rng
import copy


MEAN, STD = torch.tensor([0.7871, 0.6265, 0.7644]), torch.tensor([0.1279, 0.1786, 0.1127]) # calculated on entire dataset


class BreakHis:
    
    def __init__(self, root, train_transform, valid_transform, label='tumor_class', download=False, valid_percent=0.2, shuffle=True):
        
        self.root = root
        self.label = label
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.valid_percent = valid_percent
        self.shuffle = shuffle
        
        assert label in ['tumor_class', 'tumor_type']
        
        if download:
            self.download_data()
            
        self.ds_train, self.ds_valid, self.ds_test, self.sampler_train, self.sampler_valid, self.sampler_test = self.prepare_datasets()
        
    def download_data(self):
        url = 'http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz'
        download_and_extract_archive(url, self.root)
        self.root = self.root/'BreaKHis_v1/histology_slides/breast' # extend root directory to point to images
    
    def prepare_datasets(self):
        
        if self.label == 'tumor_type':
            
            # multiclass classification
            benign_classes    = [0,1,2,3]
            malignant_classes = [4,5,6,7]
            
            benign_types = self.root/'benign/SOB'
            malignant_types = self.root/'malignant/SOB'

            # instantiate copies of dataset
            ds_b_train = ImageFolder(benign_types, self.train_transform)
            ds_b_valid = ImageFolder(benign_types, self.valid_transform)
            ds_b_test  = ImageFolder(benign_types, self.valid_transform)

            ds_m_train = ImageFolder(malignant_types, self.train_transform)
            ds_m_valid = ImageFolder(malignant_types, self.valid_transform)
            ds_m_test  = ImageFolder(malignant_types, self.valid_transform)
            
            for ds_m in [ds_m_train, ds_m_valid, ds_m_test]:
                # offset classes
                img_paths, labels = list(zip(*ds_m.samples))
                labels = [label+4 for label in labels]
                ds_m_train.targets = labels
                ds_m_train.samples = list(zip(img_paths, labels))
                
            # split classes with equal proportion
            ds_b_train, ds_b_valid, ds_b_test = self._stratified_split(ds_b_train, ds_b_valid, ds_b_test, benign_classes)
            ds_m_train, ds_m_valid, ds_m_test = self._stratified_split(ds_m_train, ds_m_valid, ds_m_test, malignant_classes)

            # merge malignant and benign datasets
            ds_train = torch.utils.data.ConcatDataset([ds_b_train, ds_m_train])
            ds_valid = torch.utils.data.ConcatDataset([ds_b_valid, ds_m_valid])
            ds_test  = torch.utils.data.ConcatDataset([ds_b_test,  ds_m_test])

            # update targets
            ds_train.targets = ds_b_train.targets + ds_m_train.targets
            ds_valid.targets = ds_b_valid.targets + ds_m_valid.targets
            ds_test.targets  = ds_b_test.targets  + ds_m_test.targets
            
        else:
            
            # binary classification
            classes = [0,1] 
            
            ds_train = ImageFolder(self.root, self.train_transform)
            ds_valid = ImageFolder(self.root, self.valid_transform)
            ds_test  = ImageFolder(self.root, self.valid_transform)
            
            # split classes with equal proportion
            ds_train, ds_valid, ds_test = self._stratified_split(ds_train, ds_valid, ds_test, classes)
            
        
        # Use balanced sampling to handle class imbalances
        sampler_train = self._get_balanced_sampler(ds_train) 
        sampler_valid = self._get_balanced_sampler(ds_valid)
        sampler_test  = self._get_balanced_sampler(ds_test)
        
        return ds_train, ds_valid, ds_test, sampler_train, sampler_valid, sampler_test
    
    
    def _stratified_split(self, ds_train, ds_valid, ds_test, classes):
        
        X, y = list(zip(*ds_train.samples))
        
        stratify = np.repeat(classes, np.ceil(len(X)/len(classes)))[:len(X)]
        X_train, X_valid, y_train, y_valid = train_test_split(X, y,  stratify=stratify, test_size=0.4) # valid + test = 40% of train

        # split valid and test
        stratify = np.repeat(classes, np.ceil(len(X_valid)/len(classes)))[:len(X_valid)]
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid,  stratify=stratify, test_size=0.5) # valid and test are equally sized

        # update dataset samples and targets
        ds_train.samples = list(zip(X_train, y_train))
        ds_train.targets = y_train

        ds_valid.samples = list(zip(X_valid, y_valid))
        ds_valid.targets = y_valid

        ds_test.samples = list(zip(X_test,  y_test))
        ds_test.targets = y_test

        return ds_train, ds_valid, ds_test 
    
    
    def _get_balanced_sampler(self, ds):
        
        _, class_counts = np.unique(ds.targets, return_counts=True)
        n_classes = len(class_counts)
        num_samples = len(ds)
        labels = copy.copy(ds.targets)

        class_weights = [num_samples/class_counts[i] for i in range(n_classes)]
        weights = [class_weights[labels[i]] for i in range(num_samples)]
        sampler = WeightedRandomSampler(torch.tensor(weights), num_samples)
        
        return sampler
    
    
    def get_dataloaders(self, batch_size, pin_memory=True, num_workers=0):
        train_dl = DataLoader(
            self.ds_train, batch_size=batch_size, sampler=self.sampler_train,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        valid_dl = DataLoader(
            self.ds_valid, batch_size=batch_size, sampler=self.sampler_valid,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        test_dl = DataLoader(
            self.ds_test, batch_size=batch_size, sampler=self.sampler_test,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        return train_dl, valid_dl, test_dl