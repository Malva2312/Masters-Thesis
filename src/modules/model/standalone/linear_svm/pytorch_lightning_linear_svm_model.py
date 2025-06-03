from torchmetrics.functional import accuracy, auroc, precision, recall
import pytorch_lightning
import torch

from src.modules.model.standalone.linear_svm.linear_svm_model import LinearSVMModel
from src.modules.loss_functions.hinge_loss_functions import HingeLossFunction

from src.modules.features.feature_extractors import FeatureExtractorManager


class PyTorchLightningLinearSVMModel(pytorch_lightning.LightningModule):
    def __init__(self, config, experiment_execution_paths):
        super().__init__()
        self.config = config

        self.criterion = HingeLossFunction(
            criterion=self.config.svm_config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )
        
        self.feature_extractor_manager = FeatureExtractorManager(config=self.config.svm_config)

        # Use a dummy 32x32 image to determine the feature extractor output size
        with torch.no_grad():
            dummy_image = torch.zeros(1, 1, 32, 32)  # Assuming grayscale, shape: (batch, channel, height, width)
            dummy_mask = torch.zeros(1, 1, 32, 32)
            _ = self.feature_extractor_manager(dummy_image, dummy_mask)
            dummy_features = self.feature_extractor_manager.to_vector()
            self.config.svm_config.input_dim = dummy_features.shape[1]


        self.labels = None
        self.model = LinearSVMModel(input_dim=self.config.svm_config.input_dim)
        self.predicted_labels = None
        self.weighted_losses = None
        

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

    def extract_features(self, images, masks=None):
        _ = self.feature_extractor_manager(images, masks)
        return self.feature_extractor_manager.to_vector().to(self.device)

    def training_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]

        # Extract features from all extractors
        model_input = self.extract_features(data['image'], masks=data.get('mask', None))

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
        model_input = self.extract_features(data['image'], masks=data.get('mask', None))

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
        model_input = self.extract_features(data['image'], masks=data.get('mask', None))

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

    