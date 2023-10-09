import timm
import torch.nn as nn
import torch
from sklearn.metrics import mean_squared_error

class PawpularityModel(pl.LightningModule):
    def __init__(self, model_name="tf_efficientnet_b0_ns", pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(model_name=model_name, pretrained=pretrained, in_chans=3)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(128 + 12, 1)

        self.criterion = nn.BCEWithLogitsLoss()

        self.validation_step_outputs = []
        self.training_step_outputs = []

    def forward(self, input, features):
        x = self.backbone(input)
        x = self.dropout(x)

        x = torch.cat([x, features], dim=1)
        x = self.out(x)

        return x

    def training_step(self, batch, batch_indexes):
        loss, predictions, labels, rmse = self.step(batch, 'train')
        self.training_step_outputs.append({ "rmse": rmse, "loss": loss })

        return { 'loss': loss, 'predictions': predictions, 'labels': labels }

    def validation_step(self, batch, batch_indexes):
        loss, predictions, labels, rmse = self.step(batch, 'val')
        self.validation_step_outputs.append({ "rmse": rmse, "loss": loss })
        
        return { 'loss': loss, 'predictions': predictions, 'labels': labels }

    def step(self, batch, mode):
        image_ids, features, images, labels = batch
        labels = labels.float() / 100.0

        logits = self.forward(images, features).squeeze(1)
        loss = self.criterion(logits, labels) # using BCELoss to optimize models

        predictions = logits.sigmoid().detach().cpu() * 100
        labels = labels.detach().cpu() * 100
        
        rmse = mean_squared_error(predictions, labels, squared=False) # keeping track of RMSE as it is the competition metric
        rmse = torch.tensor(rmse, dtype=torch.float32)

        self.log(f'{mode}_loss', loss)
        
        return loss, predictions, labels, rmse
    
    def on_train_epoch_end(self):
        rsmes = [x["rmse"] for x in self.training_step_outputs]
        rsme = torch.stack(rsmes).mean()

        self.log(f'train_rmse', rsme, prog_bar=True)

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        rsmes = [x["rmse"] for x in self.validation_step_outputs]
        rsme = torch.stack(rsmes).mean()

        self.log(f'val_rmse', rsme, prog_bar=True)
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 20, eta_min=1e-4)

        return [optimizer], [scheduler]