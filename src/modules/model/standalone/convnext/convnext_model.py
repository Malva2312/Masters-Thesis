from torchvision import models
import torch

class ConvNextModel(torch.nn.Module):
    def __init__(self, config):
        super(ConvNextModel, self).__init__()
        convnext_name = f"convnext_{getattr(config.convnext_config, 'architecture', 'tiny')}"
        convnext_fn = getattr(models, convnext_name)
        self.model = convnext_fn(weights=None)
        self.model.classifier[2] = torch.nn.Linear(
            self.model.classifier[2].in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        # If input is grayscale, repeat channels to get 3 channels
        if model_input.shape[1] == 1:
            model_input = model_input.repeat(1, 3, 1, 1)
        return self.model(model_input)