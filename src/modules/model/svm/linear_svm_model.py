import torch

class LinearSVMModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(LinearSVMModel, self).__init__()
        self.fc = torch.nn.Linear(1024, 1)  # binary classification: one output

    def forward(self, x):
        if x.shape[-1] != self.fc.in_features:
            raise ValueError(f"Input feature dimension mismatch. Expected {self.fc.in_features}, got {x.shape[-1]}")
        return self.fc(x)  # No activation here; raw score used for hinge loss