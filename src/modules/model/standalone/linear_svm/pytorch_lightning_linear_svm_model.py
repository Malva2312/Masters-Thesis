from torchmetrics.functional import accuracy, auroc, precision, recall
import pytorch_lightning
import torch

from src.modules.model.standalone.linear_svm.linear_svm_model import LinearSVMModel
from src.modules.loss_functions.hinge_loss_functions import HingeLossFunction

# Texture features
from src.modules.features.texture.lbp import LocalBinaryPattern
from src.modules.features.texture.glcm import GrayLevelCooccurrenceMatrix

# Frequency features
from src.modules.features.frequency.fft import FastFourierTransform
from src.modules.features.frequency.gabor import GaborFeature

# Gradient features
from src.modules.features.gradient.hog import HistogramOfOrientedGradients

# Intensity features
from src.modules.features.intensity.entropy import Entropy
from src.modules.features.intensity.mean_std import MeanStd
from src.modules.features.intensity.percentiles import Percentiles

# Shape features
from src.modules.features.shape.area_perimeter import AreaPerimeter
from src.modules.features.shape.eccentricity_solidity import EccentricitySolidity


class PyTorchLightningLinearSVMModel(pytorch_lightning.LightningModule):
    def __init__(self, config, experiment_execution_paths):
        super().__init__()
        self.config = config

        self.criterion = HingeLossFunction(
            criterion=self.config.svm_config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )

        self.labels = None
        self.model = LinearSVMModel(input_dim=self.config.svm_config.input_dim)
        self.predicted_labels = None
        self.weighted_losses = None

        # Instantiate all feature extractors
        self.lbp_extractor = LocalBinaryPattern(
            P=getattr(self.config, "lbp_P", 8),
            R=getattr(self.config, "lbp_R", 1),
            method=getattr(self.config, "lbp_method", "uniform")
        )
        self.glcm_extractor = GrayLevelCooccurrenceMatrix()
        self.fft_extractor = FastFourierTransform()
        self.gabor_extractor = GaborFeature()
        self.hog_extractor = HistogramOfOrientedGradients()
        self.entropy_extractor = Entropy()
        self.mean_std_extractor = MeanStd()
        self.percentiles_extractor = Percentiles()
        self.area_perimeter_extractor = AreaPerimeter()
        self.eccentricity_solidity_extractor = EccentricitySolidity()

        self.to(torch.device(self.config.device))

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config.optimiser.type)(
            self.parameters(), **self.config.optimiser.kwargs
        )
        return optimizer

    def on_train_epoch_start(self):
        self.labels = []
        self.predicted_labels = []
        self.weighted_losses = []

    def extract_features(self, images):
        features = []
        # Each extractor should handle a batch of images (shape: [batch_size, ...])
        features.append(self.lbp_extractor(images).view(images.size(0), -1))
        features.append(self.glcm_extractor(images).view(images.size(0), -1))
        fft_features = self.fft_extractor(images)
        if isinstance(fft_features, dict):
            # Replace 'feature' with the actual key if different
            fft_features = fft_features['magnitude']  
        features.append(fft_features.view(images.size(0), -1))
        features.append(self.gabor_extractor(images).view(images.size(0), -1))
        features.append(self.hog_extractor(images).view(images.size(0), -1))
        features.append(self.entropy_extractor(images).view(images.size(0), -1))
        mean_std_features = self.mean_std_extractor(images)
        features.append(mean_std_features['mean'].view(images.size(0), -1))
        features.append(mean_std_features['std'].view(images.size(0), -1))
        features.append(self.percentiles_extractor(images).view(images.size(0), -1))
        area_perimeter_features = self.area_perimeter_extractor(images)
        features.append(area_perimeter_features['area'].view(images.size(0), -1))
        features.append(area_perimeter_features['perimeter'].view(images.size(0), -1))
        #features.append(self.eccentricity_solidity_extractor(images).view(images.size(0), -1))
        # Concatenate all features along the last dimension
        return torch.cat(features, dim=1).to(self.device)

    def training_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]

        # Extract features from all extractors
        model_input = self.extract_features(data['image'])

        model_output = self.model(model_input)

        loss = self.criterion(
            logits=model_output,
            targets=labels.to(self.device)
        )

        self.log(
            "train_loss",
            loss,
            batch_size=data['image'].shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=False
        )

        return loss

    def on_validation_epoch_start(self):
        self.labels = []
        self.predicted_labels = []
        self.weighted_losses = []

    def validation_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]

        # Extract features from all extractors
        model_input = self.extract_features(data['image'])

        model_output = self.model(model_input)
        predicted_labels = torch.sign(model_output)
        loss = self.criterion(
            logits=model_output,
            targets=labels.to(self.device)
        )

        self.labels.append(labels)
        self.predicted_labels.append(predicted_labels)
        self.weighted_losses.append(loss * data['image'].shape[0])

    def on_validation_epoch_end(self):
        labels = torch.cat(self.labels, dim=0)
        predicted_labels = torch.cat(self.predicted_labels, dim=0)

        metrics_for_logging = {
            'val_loss': (sum(self.weighted_losses) / labels.shape[0]).item(),
            'val_accuracy': accuracy(
                preds=predicted_labels,
                target=labels,
                task="binary"
            ).item(),
            'val_auroc': auroc(
                preds=predicted_labels.float(),
                target=labels.int(),
                task="binary"
            ).item(),
            'val_precision': precision(
                preds=predicted_labels,
                target=labels,
                task="binary"
            ).item(),
            'val_recall': recall(
                preds=predicted_labels,
                target=labels,
                task="binary"
            ).item()
        }
        self.log_dict(
            metrics_for_logging,
            batch_size=labels.shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=False
        )

    def on_test_epoch_start(self):
        self.labels = []
        self.predicted_labels = []

    def test_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]

        # Extract features from all extractors
        model_input = self.extract_features(data['image'])

        model_output = self.model(model_input)
        predicted_labels = torch.sign(model_output)

        self.labels.append(labels)
        self.predicted_labels.append(predicted_labels)

    def on_test_epoch_end(self):
        labels = torch.cat(self.labels, dim=0)
        predicted_labels = torch.cat(self.predicted_labels, dim=0)

        metrics_for_logging = {
            'test_accuracy': accuracy(
                preds=predicted_labels,
                target=labels,
                task="binary"
            ).item(),
            'test_auroc': auroc(
                preds=predicted_labels.float(),
                target=labels.int(),
                task="binary"
            ).item(),
            'test_precision': precision(
                preds=predicted_labels,
                target=labels,
                task="binary"
            ).item(),
            'test_recall': recall(
                preds=predicted_labels,
                target=labels,
                task="binary"
            ).item()
        }
        self.log_dict(
            metrics_for_logging,
            batch_size=labels.shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=False
        )

    