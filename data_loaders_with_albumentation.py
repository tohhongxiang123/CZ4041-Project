import numpy as np
import pandas as pd
import os

import albumentations
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningDataModule


dense_features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]

class PetFinderDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dir: str, augmentations: albumentations.Compose):
        self.ids = df["Id"].values
        if "Pawpularity" in df.keys():
            self.targets = df["Pawpularity"].values
        else:
            self.targets = [-1] * len(df)
        self.dense_features = df[dense_features].values

        image_paths = [os.path.join(dir, f"{x}.jpg") for x in df["Id"].values]
        self.image_paths = image_paths

        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        image_id = self.ids[item]

        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
            
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        features = self.dense_features[item, :]
        targets = self.targets[item]
        
        return image_id, torch.tensor(features, dtype=torch.float), torch.tensor(image, dtype=torch.float), torch.tensor(targets, dtype=torch.float)
    
class PetFinderDataModule(LightningDataModule):
    def __init__(self, 
                 df_train=None, df_val=None, df_test=None, 
                 train_images_dir=None, val_images_dir=None, test_images_dir=None, 
                 train_augmentations=None, val_augmentations=None, test_augmentations=None, 
                 batch_size=64
                ):
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        self.train_images_dir = train_images_dir
        self.val_images_dir = val_images_dir
        self.test_images_dir = test_images_dir

        self.train_augmentations = train_augmentations
        self.val_augmentations = val_augmentations
        self.test_augmentations = test_augmentations

        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(PetFinderDataset(self.df_train, self.train_images_dir, self.train_augmentations), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(PetFinderDataset(self.df_val, self.val_images_dir, self.val_augmentations), batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(PetFinderDataset(self.df_test, self.test_images_dir, self.test_augmentations), batch_size=self.batch_size, shuffle=False) 
