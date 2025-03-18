import lightning as pl
from efficientnet_pytorch import EfficientNet
import torch
from torch import optim
from torch import nn

class ProtocolEfficientNet(): # May use LungNoduleClassifier as a base class
    def __init__(self, num_classes, model_name='efficientnet-b0'):
        super().__init__()
        self.save_hyperparameters()

        self.model = EfficientNet.from_pretrained(model_name)  # Use model_name parameter
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)  # Update for num_classes

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

        labels = labels['lnm']['mean'].float().unsqueeze(1)  
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.sigmoid(logits).round()
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels['lnm']['mean'].float().unsqueeze(1)
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.sigmoid(logits).round()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Perform a single test step.

        Args:
            batch (tuple): A tuple containing the input images and corresponding labels.
            batch_idx (int): The index of the batch.

        This method performs the following steps:
        1. Extracts images and labels from the batch.
        2. Converts labels to float and adds an extra dimension.
        3. Passes the images through the model to obtain logits.
        4. Computes the loss between the logits and the labels.
        5. Calculates the predictions by applying a sigmoid function and rounding the results.
        6. Computes the accuracy by comparing the predictions with the labels.
        7. Logs the test accuracy.
        """
        images, labels = batch
        labels = labels.float().unsqueeze(1)
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.sigmoid(logits).round()
        acc = (preds == labels).float().mean()
        self.log("test_acc", acc, prog_bar=True) #And other metrics

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer