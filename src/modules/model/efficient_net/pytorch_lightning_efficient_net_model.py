from torchmetrics.functional import accuracy, auroc, precision, recall
import pytorch_lightning
import torch

from src.modules.model.efficient_net.efficient_net_model \
    import EfficientNetModel
from src.modules.loss_functions.efficient_net_loss_functions \
    import EfficientNetLossFunction


class PyTorchLightningEfficientNetModel(pytorch_lightning.LightningModule):
    def __init__(self, config, experiment_execution_paths):
        super().__init__()
        self.config = config

        self.criterion = EfficientNetLossFunction(
            config=self.config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )
        self.labels = None
        self.model = EfficientNetModel(config=self.config.model)
        self.predicted_labels = None
        self.weighted_losses = None

        self.to(torch.device(self.config.device))

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config.optimiser.type)(
            self.parameters(), **self.config.optimiser.kwargs
        )
        return optimizer

    def training_step(self, batch):
        data, labels = batch[0], batch[1]

        model_output = self.model(data['image'].to(self.device))

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

        model_output = self.model(data['image'].to(self.device))
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

        model_output = self.model(data['image'].to(self.device))
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
