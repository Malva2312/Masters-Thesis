import lightning as pl
import torch
import numpy as np
import SimpleITK as sitk
from skimage.feature import local_binary_pattern
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from radiomics import featureextractor
from modules.extractors.LBPExtractor import LBPExtractor 


class ProtocolLBP(pl.LightningModule):
    def __init__(self, radius=1, n_points=8, method='uniform', kernel='linear', C=1.0):
        super().__init__()
        self.save_hyperparameters()

        # Initialize LBP extractor
        self.extractor = LBPExtractor(radius=radius, n_points=n_points, method=method)

        # Define the pipeline
        self.model = Pipeline([
            ('lbp', self.extractor),
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=kernel, C=C))
        ])

    def step(self, batch):
        images, labels = batch
        images = images['input_image']  # List of numpy arrays (H, W)
        labels = labels['lnm']['mean']  # Typically a tensor or list
        
        # Convert 3D images to 2D by selecting the first channel
        images = [image[0, :, :] for image in images]  # Assuming the first channel is at index 0
        # Convert labels to integers
        labels = np.array(labels, dtype=int).ravel()  # Reshape to avoid DataConversionWarning

        return images, labels

    def training_step(self, batch, _batch_idx):
        images, labels = self.step(batch)
        self.model.fit(images, labels)
        return None  # Return a tensor for compatibility with PyTorch Lightning

    def test_step(self, batch, _batch_idx):
        images, labels = self.step(batch)
        predictions = self.model.predict(images)

        # Log the shape of the batch
        print("\nimages_shape", len(images))

        # Calculate accuracy
        accuracy = np.mean(predictions == labels)
        self.log("test_accuracy", accuracy, prog_bar=True)

        return None

    def validation_step(self, _batch, _batch_idx):
        pass

    def configure_optimizers(self):
        return None  # No optimizer needed for SVM
