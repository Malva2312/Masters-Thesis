from torchvision import models
import torch

class ResNetModel(torch.nn.Module):
    def __init__(self, config):
        super(ResNetModel, self).__init__()
        resnet_name = f"resnet{getattr(config.resnet_config, 'architecture', '50')}"
        resnet_fn = getattr(models, resnet_name)
        self.model = resnet_fn()
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        # If input is grayscale, repeat channels to get 3 channels
        if model_input.shape[1] == 1:
            model_input = model_input.repeat(1, 3, 1, 1)
        return self.model(model_input)