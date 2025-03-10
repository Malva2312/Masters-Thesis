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
from PIL import Image

from LungNoduleDataset import LungNoduleDataset
from LungNoduleClassifier import LungNoduleClassifier

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 pixels
    transforms.ToTensor(),  # Convert image to tensor
])

# Load dataset
image_paths = ["./dummy.jpg", "./dummy.jpg"]  # List of image file paths
labels = [1, 1]  # Corresponding labels

# Train-test split
train_paths, val_paths, train_labels, val_labels = train_test_split(
image_paths, labels, test_size=0.5, random_state=42  # Adjusted test_size to 0.5
)

# Create PyTorch datasets
train_dataset = LungNoduleDataset(train_paths, train_labels, transform=transform)
val_dataset = LungNoduleDataset(val_paths, val_labels, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

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
    image = Image.open(image_path)  # Load image using PIL
    image = transform(image).unsqueeze(0)  # Add batch dimension
    logits = model(image)
    prob = torch.sigmoid(logits).item()
    return "Malignant" if prob > 0.5 else "Benign"

# Example usage
prediction = predict(model, "./dummy.jpg")
print(f"Prediction: {prediction}")
