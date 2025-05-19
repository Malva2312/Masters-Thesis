import torch
import pytorch_lightning

from torchmetrics.functional import accuracy, auroc, precision, recall
from src.modules.model.efficient_net.efficient_net_model import EfficientNetModel
from src.modules.model.linear_svm.linear_svm_model import LinearSVMModel
from src.modules.loss_functions.efficient_net_loss_functions import EfficientNetLossFunction
from src.modules.loss_functions.hinge_loss_functions import HingeLossFunction
from src.modules.features.lbp_extractor import LocalBinaryPattern

class PyTorchLightningEffNetSVMFusionModel(pytorch_lightning.LightningModule):
    def __init__(self, config, experiment_execution_paths):
        super().__init__()
        self.config = config
        self.effnet_config = config.effnet_config
        self.svm_config = config.svm_config

        # EfficientNet
        self.effnet = EfficientNetModel(config=self.effnet_config)
        self.effnet_criterion = EfficientNetLossFunction(
            config=self.effnet_config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )

        # SVM
        self.svm = LinearSVMModel(input_dim=self.svm_config.input_dim)
        self.svm_criterion = HingeLossFunction(
            criterion=self.svm_config.criterion,
            experiment_execution_paths=experiment_execution_paths
        )
        self.lbp_extractor = LocalBinaryPattern(
            P=getattr(self.config, "lbp_P", 8),
            R=getattr(self.config, "lbp_R", 1),
            method=getattr(self.config, "lbp_method", "uniform")
        )

        self.to(torch.device(self.config.device))

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config.optimiser.type)(
            self.parameters(), **self.config.optimiser.kwargs
        )
        return optimizer

    def _get_effnet_pred(self, image):
        logits = self.effnet(image.to(self.device))
        # Binary: take sigmoid and threshold at 0.5
        probs = torch.sigmoid(logits[:, 0])
        return (probs > 0.5).long().unsqueeze(1), logits

    def _get_svm_pred(self, image):
        lbp = self.lbp_extractor(image)
        x = lbp.view(lbp.size(0), -1).to(self.device)
        logits = self.svm(x)
        # SVM: sign output, map {-1, 1} to {0, 1}
        pred = ((torch.sign(logits) + 1) // 2).long()
        return pred, logits

    def _late_fusion(self, effnet_pred, svm_pred):
        # Both are {0, 1}, fuse by majority (here, 50/50, so sum and threshold at 1)
        fused = ((effnet_pred + svm_pred) >= 1).long()
        return fused

    def training_step(self, batch, batch_idx):
        data, labels = batch[0], batch[1]

        # EfficientNet loss
        effnet_logits = self.effnet(data['image'].to(self.device))
        effnet_loss = self.effnet_criterion(
            logits=effnet_logits,
            targets=labels.to(self.device)
        )

        # SVM loss
        lbp = self.lbp_extractor(data['image'])
        svm_input = lbp.view(lbp.size(0), -1).to(self.device)
        svm_logits = self.svm(svm_input)
        # SVM expects targets in {-1, 1}
        svm_targets = labels.clone().to(self.device)
        svm_targets[svm_targets == 0] = -1
        svm_loss = self.svm_criterion(
            logits=svm_logits,
            targets=svm_targets
        )

        # Average the losses
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

        # EfficientNet
        effnet_pred, effnet_logits = self._get_effnet_pred(data['image'])
        effnet_loss = self.effnet_criterion(
            logits=effnet_logits,
            targets=labels.to(self.device)
        )

        # SVM
        svm_pred, svm_logits = self._get_svm_pred(data['image'])
        svm_targets = labels.clone().to(self.device)
        svm_targets[svm_targets == 0] = -1
        svm_loss = self.svm_criterion(
            logits=svm_logits,
            targets=svm_targets
        )

        # Late fusion
        fused_pred = self._late_fusion(effnet_pred, svm_pred)

        # Average the losses
        loss = 0.5 * effnet_loss + 0.5 * svm_loss

        self.labels.append(labels)
        self.preds.append(fused_pred)
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

        effnet_pred, _ = self._get_effnet_pred(data['image'])
        svm_pred, _ = self._get_svm_pred(data['image'])
        fused_pred = self._late_fusion(effnet_pred, svm_pred)

        self.labels.append(labels)
        self.preds.append(fused_pred)

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