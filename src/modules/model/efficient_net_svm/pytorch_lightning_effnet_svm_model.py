import torch
import pytorch_lightning

from .effnet_svm_model import EfficientNetSVMFusedModel

class LightningEffNetSVM(pytorch_lightning.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.model = EfficientNetSVMFusedModel(config)
        self.lr = getattr(config, "lr", 1e-3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        effnet_logits, svm_logits = self(x)
        total_loss, ce_loss, hinge_loss = self.model.compute_losses(effnet_logits, svm_logits, y)
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/ce_loss", ce_loss)
        self.log("train/hinge_loss", hinge_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        effnet_logits, svm_logits = self(x)
        total_loss, ce_loss, hinge_loss = self.model.compute_losses(effnet_logits, svm_logits, y)
        self.log("val/total_loss", total_loss, prog_bar=True)
        self.log("val/ce_loss", ce_loss)
        self.log("val/hinge_loss", hinge_loss)
        preds = torch.argmax(effnet_logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val/acc", acc, prog_bar=True)
        return {"val_loss": total_loss, "val_acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)