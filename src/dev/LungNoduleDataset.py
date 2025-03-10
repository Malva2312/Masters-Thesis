import torch
from torch.utils.data import Dataset
from PIL import Image

class LungNoduleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # image = torch.load(self.image_paths[idx])  # Load preprocessed image tensor
        image = Image.open(self.image_paths[idx])  # Load image using PIL
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label