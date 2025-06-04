import torch
import numpy as np
from skimage.feature import hog

class HOGFeatureExtractor:
    def __init__(self, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.orientations = orientations

    def __call__(self, images, masks=None):
        """
        images: torch.Tensor of shape (N, H, W) or (H, W)
        masks: torch.Tensor of shape (N, H, W) or (H, W) or None
        Returns: dict with key 'hog' and value as torch.Tensor of features
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        if masks is not None and isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        # Handle single image
        if images.ndim == 2:
            images = images[None, ...]
            if masks is not None and masks.ndim == 2:
                masks = masks[None, ...]

        features = []
        for idx, img in enumerate(images):
            if masks is not None:
                mask = masks[idx]
                img = img * mask  # Apply mask
            # skimage hog expects float images
            img = img.astype(np.float32)
            hog_feat = hog(
                img,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm='L2-Hys',
                visualize=False,
                feature_vector=True
            )
            features.append(hog_feat)
        features = np.stack(features, axis=0)
        return {'hog': features}