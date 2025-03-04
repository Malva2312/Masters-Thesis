import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet

class LungNoduleClassifier(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Load EfficientNet and modify final layer
        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        # self.model = FusionModel(protocol = 1)  # Custom model for fusion
        self.model._fc = nn.Linear(self.model._fc.in_features, 1)  # Binary classification

        self.loss_fn = nn.BCEWithLogitsLoss()  # For binary classification
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float().unsqueeze(1)  # Ensure shape compatibility
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.sigmoid(logits).round()
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float().unsqueeze(1)
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.sigmoid(logits).round()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
