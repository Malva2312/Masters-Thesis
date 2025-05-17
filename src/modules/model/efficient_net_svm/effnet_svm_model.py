import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class EfficientNetSVMFusedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # EfficientNet backbone
        self.efficient_net = efficientnet_b0()
        self.efficient_net.classifier[1] = nn.Linear(
            self.efficient_net.classifier[1].in_features,
            config.number_of_classes
        )
        # SVM head (linear)
        self.svm_head = nn.Linear(
            self.efficient_net.classifier[1].in_features,
            config.number_of_classes
        )
        # Loss weights
        self.alpha = getattr(config, "alpha", 0.5)  # weight for classification loss
        self.beta = getattr(config, "beta", 0.5)    # weight for SVM loss

    def forward(self, x):
        # Extract features
        features = self.efficient_net.features(x.repeat(1, 3, 1, 1))
        pooled = self.efficient_net.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        # EfficientNet classifier output
        effnet_logits = self.efficient_net.classifier[1](pooled)
        # SVM output
        svm_logits = self.svm_head(pooled)
        return effnet_logits, svm_logits

    def compute_losses(self, effnet_logits, svm_logits, targets):
        # Classification loss (cross-entropy)
        ce_loss = nn.CrossEntropyLoss()(effnet_logits, targets)
        # SVM hinge loss (multi-class)
        targets_one_hot = torch.zeros_like(svm_logits).scatter_(1, targets.unsqueeze(1), 1)
        hinge_loss = torch.mean(
            torch.clamp(1 - svm_logits * (2 * targets_one_hot - 1), min=0)
        )
        # Weighted sum
        total_loss = self.alpha * ce_loss + self.beta * hinge_loss
        return total_loss, ce_loss, hinge_loss