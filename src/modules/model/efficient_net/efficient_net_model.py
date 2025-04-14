from torchvision.models import efficientnet_b0
import torch


class EfficientNetModel(torch.nn.Module):
    def __init__(self, config):
        super(EfficientNetModel, self).__init__()
        self.model = efficientnet_b0()
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        model_output = self.model(model_input.repeat(1, 3, 1, 1))
        return model_output
