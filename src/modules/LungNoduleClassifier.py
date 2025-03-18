import lightning as pl
from torch import optim, nn
import torch

class LungNoduleClassifier(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, 64), 
            nn.ReLU(), 
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32 * 32),
            nn.Unflatten(1, (1, 32, 32))
        )
        
    def forward(self, x):
        x = x['input_image']  # Assuming 'input_image' is the key for the tensor in the dict
        return self.encoder(x)
    
    def training_step(self, batch, batch_idx):
        """
        Defines the training step for the model.
        Args:
            batch (tuple): A tuple containing the input data and labels. 
                           The input data is a dictionary with the key 'input_image' 
                           which maps to a list of tensor images (one image for each channel).
            batch_idx (int): The index of the batch.
        Returns:
            torch.Tensor: The loss value computed for the batch.
        Notes:
            - The method extracts the images from the batch and reshapes them.
            - It then passes the images through the encoder and decoder to compute the reconstructed images.
            - The mean squared error (MSE) loss is calculated between the original and reconstructed images.
            - The loss is logged for monitoring purposes.
        """
        # training_step defines the train loop.
        # it is independent of forward
        images, labels = batch
        labels = labels['lnm']['mean'].float().unsqueeze(1)

        logits = self(images)

        images = images['input_image']  # Assuming 'input_image' is the key for the tensor in the dict
        images = images.view(images.size(0), -1)

        z = self.encoder(images)
        images_hat = self.decoder(z)

        # Ensure the dimensions match correctly
        images = images.view(images.size(0), 1, 32, 32)
        images_hat = images_hat.view(images.size(0), 1, 32, 32)

        loss = nn.functional.mse_loss(images_hat, images)
        preds = torch.sigmoid(logits).round()
        acc = (preds == labels).float().mean()

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
