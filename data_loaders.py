import os

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision.io import read_image
from pytorch_lightning import LightningDataModule


# Dataset
columns = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']

class PetFinderDataset(Dataset):
    def __init__(self, df, image_dir, image_size=224):
        self.image_ids = df["Id"].values
        self.features = df[columns].values
        self.labels = None

        if "Pawpularity" in df.keys():
            self.labels = df["Pawpularity"].values

        self.image_dir = image_dir
        self.transform = T.Resize([image_size, image_size], antialias=True)
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        features = self.features[idx]

        image_id = self.image_ids[idx]
        image = read_image(os.path.join(self.image_dir, image_id + '.jpg'))
        image = self.transform(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            return image_id, features, image, label
        
        return image_id, features, image

# Data Module
class PetFinderDataModule(LightningDataModule):
    def __init__(self, df_train=None, df_val=None, df_test=None, train_dir=None, val_dir=None, test_dir=None, batch_size=64, image_size=224):
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        self.batch_size = batch_size
        self.image_size = image_size

    def train_dataloader(self):
        return DataLoader(PetFinderDataset(self.df_train, self.train_dir, self.image_size), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(PetFinderDataset(self.df_val, self.val_dir, self.image_size), batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(PetFinderDataset(self.df_test, self.test_dir, self.image_size), batch_size=self.batch_size, shuffle=False) 
