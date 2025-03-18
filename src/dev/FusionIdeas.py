import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# Define Mid-Fusion Module
class MidFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(MidFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)
        self.fc_fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x1, x2):
        x1 = self.activation(self.fc1(x1))
        x2 = self.activation(self.fc2(x2))
        fused = torch.cat((x1, x2), dim=1)
        return self.activation(self.fc_fuse(fused))

# Define Late-Fusion Module
class LateFusion(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(LateFusion, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, num_classes)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.fc_fuse = nn.Linear(num_classes * 2, num_classes)
    
    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        fused = torch.cat((x1, x2), dim=1)
        return self.fc_fuse(fused)

# Define Full Model with Nested Fusion
class LungNoduleFusionModel(pl.LightningModule):
    def __init__(self, input_dim1, input_dim2, hidden_dim, num_classes, lr=1e-3):
        super(LungNoduleFusionModel, self).__init__()
        self.mid_fusion = MidFusion(input_dim1, input_dim2, hidden_dim)
        self.late_fusion = LateFusion(hidden_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
    
    def forward(self, x1, x2):
        mid_out = self.mid_fusion(x1, x2)
        output = self.late_fusion(mid_out, mid_out)  # Nested Fusion
        return output
    
    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        logits = self.forward(x1, x2)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Example Dataset
class DummyLungNoduleDataset(Dataset):
    def __init__(self, num_samples=1000, input_dim1=128, input_dim2=128, num_classes=2):
        self.x1 = torch.randn(num_samples, input_dim1)
        self.x2 = torch.randn(num_samples, input_dim2)
        self.y = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]

# Training Script
if __name__ == "__main__":
    dataset = DummyLungNoduleDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = LungNoduleFusionModel(input_dim1=128, input_dim2=128, hidden_dim=64, num_classes=2)
    trainer = pl.Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(model, dataloader)
