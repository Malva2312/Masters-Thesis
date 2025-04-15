import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearSVMModel(nn.Module):
    def _init_(self, input_dim):
        super(LinearSVMModel, self)._init_()
        self.fc = nn.Linear(input_dim, 1)  # binary classification: one output

    def forward(self, x):
        return self.fc(x)  # No activation here; raw score used for hinge loss