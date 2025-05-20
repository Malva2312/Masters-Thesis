from torchvision import models
import torch

class GoogLeNetModel(torch.nn.Module):
    def __init__(self, config):
        super(GoogLeNetModel, self).__init__()
        self.model = models.googlenet(aux_logits=False)
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        if model_input.shape[1] == 1:
            model_input = model_input.repeat(1, 3, 1, 1)
        return self.model(model_input)