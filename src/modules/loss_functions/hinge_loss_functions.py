from torch.nn.functional import binary_cross_entropy_with_logits
import torch

from src.modules.data.metadataframe.metadataframe import MetadataFrame

class HingeLossFunction(torch.nn.Module):
    def __init__(self, criterion=None, experiment_execution_paths):
        super(HingeLossFunction, self).__init__()
        self.criterion = criterion

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        loss = self.hinge_loss(logits, targets)
        return loss

    def hinge_loss(self, outputs, labels, margin=1.0):
        # labels should be -1 or 1
        labels = labels.view(-1, 1)
        losses = torch.clamp(margin - outputs * labels, min=0)
        return losses.mean()