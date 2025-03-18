import lightning as pl
from torch import optim, nn
import torch

class LungNoduleClassifier(pl.LightningModule):
    def __init__(self, model=None, num_classes=1, num_channels=1):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.model = self.encoder if model is None else model


        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)
    
    def step(self, batch):
        images, labels = batch
        images = images['input_image']
        labels = labels['lnm']['mean']
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.sigmoid(logits).round()
        acc = (preds == labels).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
