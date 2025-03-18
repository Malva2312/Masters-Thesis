import lightning as pl
from torch import optim, nn
import torch
from radiomics import featureextractor
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np


class ProtocolRadiomics(pl.LightningModule):
    def __init__(self, num_classes=1, num_channels=1):
        super().__init__()
        self.save_hyperparameters()

        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = nn.Sequential(
            nn.Linear(num_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def extract_features(self, image):
        # Assuming image is a numpy array
        features = self.extractor.execute(image, maskFilepath=None)
        features = torch.tensor([self.extract_features(img) for img in image], dtype=torch.float32)
        return self.model(features)

    def forward(self, images):
        features = np.array([self.extract_features(img) for img in images])
        return self.model.predict_proba(features)[:, 1]
    
    def step(self, batch, batch_idx):
        images, labels = batch
        images = images['input_image']
        labels = labels['lnm']['mean']
        logits = torch.tensor(self.forward(images), dtype=torch.float32)
        loss = self.loss_fn(logits, labels)
        preds = torch.sigmoid(logits).round()
        acc = (preds == labels).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
        loss, acc = self.step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
