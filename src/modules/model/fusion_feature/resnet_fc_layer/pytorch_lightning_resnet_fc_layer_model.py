from torchmetrics.functional import accuracy, auroc, precision, recall
import pytorch_lightning
import torch

from src.modules.model.fusion_feature.resnet_fc_layer.resnet_fc_layer_model import ResNet_FC_Layer_Model
from src.modules.loss_functions.resnet_loss_functions import ResNetLossFunction
from src.modules.features.feature_extractors import FeatureExtractorManager

class PyTorchLightningResNetFCLayerModel(pytorch_lightning.LightningModule):
    def __init__(self, config, experiment_execution_paths):
        super().__init__()
        self.config = config

        self.criterion = ResNetLossFunction(
            config=self.config.resnet_config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )

        self.model = ResNet_FC_Layer_Model(config=self.config)
        self.feature_extractor_manager = FeatureExtractorManager(config=config.resnet_config)

        self.model.fusion_feature_dim = self.feature_extractor_manager.get_total_feature_dim()
        self.model.update_fusion_dim(self.model.fusion_feature_dim)

        self.labels = None
        self.predicted_labels = None
        self.weighted_losses = None

        self.to(torch.device(self.config.device))

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config.optimiser.type)(
            self.parameters(), **self.config.optimiser.kwargs
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]

        extracted_features = self.feature_extractor_manager(
            images=data['image'].to(self.device)
        )

        model_output = self.model(
            model_input=data['image'].to(self.device),
            aux_input=extracted_features
        )
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

    def validation_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]
        extracted_features = self.feature_extractor_manager(
            images=data['image'].to(self.device)
        )
        model_output = self.model(
            model_input=data['image'].to(self.device),
            aux_input=extracted_features
        )
        predicted_labels = torch.argmax(model_output, dim=1, keepdim=True)
        loss = self.criterion(
            logits=model_output,
            targets=labels.to(self.device)
        )
        self.labels.append(labels)
        self.predicted_labels.append(predicted_labels)
        self.weighted_losses.append(loss * data['image'].shape[0])

    def test_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]
        extracted_features = self.feature_extractor_manager(
            images=data['image'].to(self.device)
        )
        model_output = self.model(
            model_input=data['image'].to(self.device),
            aux_input=extracted_features
        )
        predicted_labels = torch.argmax(model_output, dim=1, keepdim=True)
        self.labels.append(labels)
        self.predicted_labels.append(predicted_labels)
        data, labels = batch[0], batch[1]

        extracted_features = self.feature_extractor_manager(
            images=data['image'].to(self.device)
        )
        model_output = self.model(
            model_input=data['image'].to(self.device),
            aux_input=extracted_features
        )
        loss = self.criterion(
            logits=model_output,
            targets=labels.to(self.device)
        )
        self.log(
            batch_size=data['image'].shape[0],
            name="train_loss",
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            value=loss
        )
        return loss

    def on_validation_epoch_start(self):
        self.labels = []
        self.predicted_labels = []
        self.weighted_losses = []

    def validation_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]

        extracted_features = self.feature_extractor_manager(
            images=data['image'].to(self.device)
        )
        model_output = self.model(
            model_input=data['image'].to(self.device),
            aux_input=extracted_features
        )
        predicted_labels = torch.argmax(model_output, dim=1, keepdim=True)
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
        extracted_features = self.feature_extractor_manager(
            images=data['image'].to(self.device)
        )
        model_output = self.model(
            model_input=data['image'].to(self.device),
            aux_input=extracted_features
        )
        predicted_labels = torch.argmax(model_output, dim=1, keepdim=True)
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