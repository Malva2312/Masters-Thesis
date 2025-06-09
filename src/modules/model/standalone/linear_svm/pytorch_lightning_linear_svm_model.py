from torchmetrics.functional import accuracy, auroc, precision, recall
import pytorch_lightning
import torch

from src.modules.model.standalone.linear_svm.linear_svm_model import LinearSVMModel
from src.modules.loss_functions.hinge_loss_functions import HingeLossFunction

from src.modules.features.feature_extractors import FeatureExtractorManager
import functools
import operator


class PyTorchLightningLinearSVMModel(pytorch_lightning.LightningModule):
    def __init__(self, config, experiment_execution_paths):
        super().__init__()
        self.config = config
        self.extractors = self.config.svm_config.extractors

        self.criterion = HingeLossFunction(
            criterion=self.config.svm_config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )


        self.features_extractor = FeatureExtractorManager()
        # Example: Call with dummy image to initialize feature extractors (if needed)
        dummy_data =  torch.zeros(1, 32, 32)
        dummy_mask = torch.zeros(1, 32, 32)  # Dummy mask if needed
        self.features_extractor(dummy_data, dummy_mask)
        self.features_names = self.config.svm_config.get('extractors', list(self.features_extractor.feature_dims.keys()))

        def prod(iterable):
            return functools.reduce(operator.mul, iterable, 1)

        self.input_dim = sum(
            prod(self.features_extractor.feature_dims[key]) for key in self.features_extractor.feature_dims.keys() if key in self.extractors
        )

        self.model = LinearSVMModel(input_dim=self.input_dim)

        self.labels = None
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

    def training_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]

        model_input = self._prepare_model_input(data)

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
        model_input = self._prepare_model_input(data)

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
        model_input = self._prepare_model_input(data)

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

    def _prepare_model_input(self, data):
        # Move all relevant features to the correct device and concatenate them
        features = []

        for key in self.features_names:
            feature = data[key]
            if isinstance(feature, torch.Tensor):
                features.append(feature.flatten(start_dim=1) if feature.dim() > 1 else feature.unsqueeze(1))
            elif isinstance(feature, list):
                tensor_feature = torch.tensor(feature).float().to(self.device)
                features.append(tensor_feature.flatten().unsqueeze(0) if tensor_feature.dim() == 1 else tensor_feature.flatten(start_dim=1))
            else:
                features.append(torch.tensor([feature], device=self.device).float().unsqueeze(0))
            
        model_input = torch.cat(features, dim=1)
        return model_input