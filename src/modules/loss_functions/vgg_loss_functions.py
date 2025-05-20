import torch

class VGGLossFunction(torch.nn.Module):
    def __init__(self, config, experiment_execution_paths=None):
        super().__init__()
        self.apply_weights = getattr(config, "apply_weights", False)
        self.device = getattr(config, "device", "cpu")
        self.class_weights = None
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)