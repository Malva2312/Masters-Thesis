import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from monai.transforms import Compose, Resize, EnsureType
from monai.data import ImageDataset

from LungNoduleDataset import LungNoduleDataset
from LungNoduleClassifier import LungNoduleClassifier

# Define transforms
transform = Compose([
    Resize((224, 224)),  # Resize to match EfficientNet input size
    EnsureType(),
])

# Load dataset
image_paths = ["./data/dummy.png", "./data/dummy.png"]  # List of preprocessed image file paths
labels = [1,1]  # Corresponding labels

# Train-test split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.5, random_state=42  # Adjusted test_size to 0.5
)

# Create PyTorch datasets
train_dataset = LungNoduleDataset(train_paths, train_labels, transform=transform)
val_dataset = LungNoduleDataset(val_paths, val_labels, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Trainer configuration
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    log_every_n_steps=10
)

# Initialize model
model = LungNoduleClassifier(lr=1e-4)

# Train the model
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def predict(model, image_path):
    model.eval()
    image = torch.load(image_path)  # Load preprocessed image tensor
    image = transform(image).unsqueeze(0)  # Add batch dimension
    logits = model(image)
    prob = torch.sigmoid(logits).item()
    return "Malignant" if prob > 0.5 else "Benign"

# Example usage
prediction = predict(model, "path/to/sample_ct_scan.pt")
print(f"Prediction: {prediction}")

# Example usage with dummy image
dummy_image_path = "dummy.png"
prediction = predict(model, dummy_image_path)
print(f"Prediction for dummy image: {prediction}")

# Example usage with dummy image for training
dummy_image_paths = ["dummy.png"] * 10  # List of dummy image paths
dummy_labels = [0] * 10  # Corresponding dummy labels

# Create dummy dataset
dummy_dataset = LungNoduleDataset(dummy_image_paths, dummy_labels, transform=transform)
dummy_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)

# Train the model with dummy data
trainer.fit(model, train_dataloaders=dummy_loader, val_dataloaders=dummy_loader)
