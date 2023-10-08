import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision.io import read_image
import timm
from timm import create_model

import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule
from transforms import train_transforms, test_transforms

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error

import glob
import gc

from data_loaders import columns

class PawpularityModel(pl.LightningModule):
    def __init__(self, model_name="swin_large_patch4_window7_224", pretrained=True):
        super().__init__()
        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.backbone = create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=3).to('cuda')
        self.dropout = nn.Dropout(0.2)
        num_features = self.backbone.num_features

        self.fc = nn.Sequential(
            nn.Linear(num_features + len(columns), int(num_features / 2)),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(int(num_features / 2), int(num_features / 4)),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(int(num_features / 4), 1)
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        
    def forward(self, input, features):
        x = self.backbone(input)
        x = self.dropout(x)
        x = torch.cat([x, features], dim=1)
        x = self.fc(x)

        return x

    def step(self, batch, mode):
        image_ids, features, images, labels = batch
        labels = labels.float() / 100.0

        images = self.train_transforms(images) if mode == "train" else self.test_transforms(images)
        logits = self.forward(images, features).squeeze(1)
        loss = self.criterion(logits, labels)

        predictions = logits.sigmoid().detach().cpu() * 100
        labels = labels.detach().cpu() * 100

        self.log(f'{mode}_loss', loss)
        
        return loss, predictions, labels

    def training_step(self, batch, batch_indexes):
        loss, predictions, labels = self.step(batch, 'train')
        self.training_step_outputs.append(loss)
        return { 'loss': loss, 'predictions': predictions, 'labels': labels }

    def validation_step(self, batch, batch_indexes):
        loss, predictions, labels = self.step(batch, 'val')
        self.validation_step_outputs.append(loss)
        return { 'loss': loss, 'predictions': predictions, 'labels': labels }
    
    def on_train_epoch_end(self):
        print(f"Training loss: {torch.stack(self.training_step_outputs).mean()}")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        print(f"Validation loss: {torch.stack(self.validation_step_outputs).mean()}")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 20, eta_min=1e-4)

        return [optimizer], [scheduler]