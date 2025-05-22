from torchvision import models
import torch

class AlexNetModel(torch.nn.Module):
    def __init__(self, config):
        super(AlexNetModel, self).__init__()
        self.model = models.alexnet()
        self.model.classifier[6] = torch.nn.Linear(
            self.model.classifier[6].in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        if model_input.shape[1] == 1:
            model_input = model_input.repeat(1, 3, 1, 1)
        # Resize input if spatial dimensions are too small
        if model_input.shape[2] < 224 or model_input.shape[3] < 224:
            model_input = torch.nn.functional.interpolate(model_input, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(model_input)