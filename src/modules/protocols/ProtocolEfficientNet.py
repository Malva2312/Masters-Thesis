import lightning as pl
from efficientnet_pytorch import EfficientNet
import torch
from torch import optim, nn

class ProtocolEfficientNet(pl.LightningModule):
    def __init__(self, num_classes, model_name='efficientnet-b0'):
        super().__init__()
        self.save_hyperparameters()
        self.model = EfficientNet.from_pretrained(model_name)
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)
    
    def step(self, batch):
        images, labels = batch
        labels = labels['lnm']['mean'].float().unsqueeze(1)
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
