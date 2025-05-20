import torch
import pytorch_lightning
from torchmetrics.functional import accuracy, auroc, precision, recall

from src.modules.model.fusion_decision.effnet_svm.effnet_svm_fusion_model import EffNetSVMFusionModule

from src.modules.loss_functions.efficient_net_loss_functions import EfficientNetLossFunction
from src.modules.loss_functions.hinge_loss_functions import HingeLossFunction

class PyTorchLightningEffNetSVMFusionModel(pytorch_lightning.LightningModule):
    def __init__(self, config, experiment_execution_paths):
        super().__init__()
        self.config = config
        self.model = EffNetSVMFusionModule(config)
        self.effnet_criterion = EfficientNetLossFunction(
            config=config.effnet_config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )
        self.svm_criterion = HingeLossFunction(
            criterion=config.svm_config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )
        self.to(torch.device(config.device))

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config.optimiser.type)(
            self.parameters(), **self.config.optimiser.kwargs
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]
        outputs = self.model(data['image'])

        effnet_loss = self.effnet_criterion(
            logits=outputs["effnet_logits"],
            targets=labels.to(self.device)
        )

        svm_targets = labels.clone().to(self.device)
        svm_targets[svm_targets == 0] = -1
        svm_loss = self.svm_criterion(
            logits=outputs["svm_logits"],
            targets=svm_targets
        )

        loss = 0.5 * effnet_loss + 0.5 * svm_loss
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
        self.preds = []
        self.weighted_losses = []

    def validation_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]
        outputs = self.model(data['image'])

        effnet_loss = self.effnet_criterion(
            logits=outputs["effnet_logits"],
            targets=labels.to(self.device)
        )

        svm_targets = labels.clone().to(self.device)
        svm_targets[svm_targets == 0] = -1
        svm_loss = self.svm_criterion(
            logits=outputs["svm_logits"],
            targets=svm_targets
        )

        loss = 0.5 * effnet_loss + 0.5 * svm_loss

        self.labels.append(labels)
        self.preds.append(outputs["fused_pred"])
        self.weighted_losses.append(loss * data['image'].shape[0])

    def on_validation_epoch_end(self):
        labels = torch.cat(self.labels, dim=0)
        preds = torch.cat(self.preds, dim=0)

        metrics_for_logging = {
            'val_loss': (sum(self.weighted_losses) / labels.shape[0]).item(),
            'val_accuracy': accuracy(
                preds=preds,
                target=labels,
                task="binary"
            ).item(),
            'val_auroc': auroc(
                preds=preds.float(),
                target=labels.int(),
                task="binary"
            ).item(),
            'val_precision': precision(
                preds=preds,
                target=labels,
                task="binary"
            ).item(),
            'val_recall': recall(
                preds=preds,
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
        self.preds = []

    def test_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]
        outputs = self.model(data['image'])
        self.labels.append(labels)
        self.preds.append(outputs["fused_pred"])

    def on_test_epoch_end(self):
        labels = torch.cat(self.labels, dim=0)
        preds = torch.cat(self.preds, dim=0)

        metrics_for_logging = {
            'test_accuracy': accuracy(
                preds=preds,
                target=labels,
                task="binary"
            ).item(),
            'test_auroc': auroc(
                preds=preds.float(),
                target=labels.int(),
                task="binary"
            ).item(),
            'test_precision': precision(
                preds=preds,
                target=labels,
                task="binary"
            ).item(),
            'test_recall': recall(
                preds=preds,
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
