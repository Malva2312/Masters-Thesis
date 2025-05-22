import torch
import numpy as np
from skimage.feature import graycomatrix, graycoprops

class GrayLevelCooccurrenceMatrix:
    def __init__(self, distances=[1], angles=[0], levels=256, symmetric=True, normed=True, props=['contrast']):
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.symmetric = symmetric
        self.normed = normed
        self.props = props

    def _apply_glcm_to_image(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        # Ensure image is in the correct range and type
        image = np.clip(image, 0, self.levels - 1).astype(np.uint8)
        if mask is not None:
            mask = mask.astype(bool)
            if mask.sum() == 0:
                # No valid pixels, return zeros
                return np.zeros((len(self.props), len(self.distances), len(self.angles)), dtype=np.float32)
            # Crop to bounding box of mask to minimize bias
            coords = np.argwhere(mask)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            image = image[y0:y1, x0:x1]
            mask = mask[y0:y1, x0:x1]
            # Set pixels outside mask to 0 (or any constant)
            image = np.where(mask, image, 0)
        glcm = graycomatrix(
            image,
            distances=self.distances,
            angles=self.angles,
            levels=self.levels,
            symmetric=self.symmetric,
            normed=self.normed
        )
        features = []
        for prop in self.props:
            feat = graycoprops(glcm, prop)
            features.append(feat)
        return np.stack(features, axis=0)

    def __call__(self, images: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): (H, W), (1, H, W), or (B, H, W)/(B, 1, H, W)
            masks (torch.Tensor, optional): Same shape as images (without channel dim), or None.
        """
        original_shape = images.shape
        device = images.device
        if len(original_shape) == 2:
            images = images.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            if masks is not None:
                masks = masks.unsqueeze(0)
        elif len(original_shape) == 3:
            images = images.unsqueeze(1)  # (B, 1, H, W)
            if masks is not None and masks.dim() == 2:
                masks = masks.unsqueeze(0)
        elif len(original_shape) == 4 and images.shape[1] != 1:
            raise ValueError("GLCM expects grayscale images with 1 channel.")

        glcm_results = []
        batch_size = images.shape[0]
        for i in range(batch_size):
            img = images[i, 0]
            img_np = img.cpu().numpy()
            mask_np = None
            if masks is not None:
                mask_np = masks[i].cpu().numpy()
            glcm_np = self._apply_glcm_to_image(img_np, mask_np)
            glcm_tensor = torch.from_numpy(glcm_np).float()
            glcm_results.append(glcm_tensor)

        output = torch.stack(glcm_results).to(device)
        if len(original_shape) == 2:
            return output[0]
        elif len(original_shape) == 3:
            return output
        elif len(original_shape) == 4:
            return output
