from torchvision import models
import torch

class VGGModel(torch.nn.Module):
    def __init__(self, config):
        super(VGGModel, self).__init__()
        vgg_name = f"vgg{getattr(config.vgg_config, 'architecture', '16')}"
        vgg_fn = getattr(models, vgg_name)
        self.model = vgg_fn()
        self.model.classifier[6] = torch.nn.Linear(
            self.model.classifier[6].in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        if model_input.shape[1] == 1:
            model_input = model_input.repeat(1, 3, 1, 1)
        return self.model(model_input)