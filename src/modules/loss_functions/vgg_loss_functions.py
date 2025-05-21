import torch
from src.modules.data.metadataframe.metadataframe import MetadataFrame

class VGGLossFunction(torch.nn.Module):
    def __init__(self, config, experiment_execution_paths=None):
        super().__init__()
        self.config = config
        self.class_weights = self._get_label_weights(experiment_execution_paths)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets.squeeze().long())

    def _get_label_weights(self, experiment_execution_paths):
        if getattr(self.config, "apply_weights", False):
            metadataframe = MetadataFrame(
                config=self.config.metadataframe,
                experiment_execution_paths=experiment_execution_paths
            )
            lung_nodule_metadataframe = metadataframe.get_lung_nodule_metadataframe()
            label_counts = lung_nodule_metadataframe['label'].value_counts().sort_index()
            label_weights = torch.tensor(
                (label_counts.min() / label_counts).tolist()
            ).to(getattr(self.config, "device", "cpu"))
        else:
            label_weights = torch.tensor([1.0, 1.0]).to(getattr(self.config, "device", "cpu"))
        return label_weights