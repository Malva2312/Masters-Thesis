from torchvision import models
import torch

class EfficientNetModel(torch.nn.Module):
    def __init__(self, config):
        super(EfficientNetModel, self).__init__()
        effnet_name = f"efficientnet_{config.effnet_config.architecture}"
        if hasattr(config.effnet_config, "architecture"):
            effnet_arch = config.effnet_config.architecture
        else:
            effnet_arch = "b0"
        effnet_name = f"efficientnet_{effnet_arch}"
        effnet_fn = getattr(models, effnet_name)
        self.model = effnet_fn()
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        model_output = self.model(model_input.repeat(1, 3, 1, 1))
        return model_output
