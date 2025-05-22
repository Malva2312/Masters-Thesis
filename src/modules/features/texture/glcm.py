import torch
import numpy as np
from skimage.feature import graycomatrix, greycoprops

class GrayLevelCooccurrenceMatrix:
    def __init__(self, distances=[1], angles=[0], levels=256, symmetric=True, normed=True, props=['contrast']):
        """
        Initialize the GLCM transformer.

        Args:
            distances (list): List of pixel pair distance offsets.
            angles (list): List of pixel pair angles in radians.
            levels (int): The input image should contain integers in [0, levels-1].
            symmetric (bool): If True, the GLCM is symmetric.
            normed (bool): If True, normalize each GLCM.
            props (list): List of GLCM properties to extract (e.g., 'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation').
        """
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.symmetric = symmetric
        self.normed = normed
        self.props = props

    def _apply_glcm_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies GLCM to a single grayscale image and extracts properties.

        Args:
            image (np.ndarray): Grayscale image.

        Returns:
            np.ndarray: Array of GLCM properties with shape (len(props), len(distances), len(angles)).
        """
        # Ensure image is in the correct range and type
        image = np.clip(image, 0, self.levels - 1).astype(np.uint8)
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
            feat = greycoprops(glcm, prop)  # shape: (len(distances), len(angles))
            features.append(feat)
        # Stack features along first axis: (len(props), len(distances), len(angles))
        return np.stack(features, axis=0)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Applies GLCM to a single image or batch of images.

        Args:
            images (torch.Tensor): Tensor of shape (H, W), (1, H, W), or (B, H, W)/(B, 1, H, W).

        Returns:
            torch.Tensor: Tensor of GLCM features, shape (B, len(props), len(distances), len(angles)).
        """
        original_shape = images.shape
        if len(original_shape) == 2:
            images = images.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif len(original_shape) == 3:
            images = images.unsqueeze(1)  # (B, 1, H, W)
        elif len(original_shape) == 4 and images.shape[1] != 1:
            raise ValueError("GLCM expects grayscale images with 1 channel.")

        glcm_results = []
        for img in images:
            img_np = img[0].cpu().numpy()  # Take the single channel
            glcm_np = self._apply_glcm_to_image(img_np)
            glcm_tensor = torch.from_numpy(glcm_np).float()
            glcm_results.append(glcm_tensor)

        output = torch.stack(glcm_results)  # (B, len(props), len(distances), len(angles))

        if len(original_shape) == 2:
            return output[0]
        elif len(original_shape) == 3:
            return output
        elif len(original_shape) == 4:
            return output  # (B, len(props), len(distances), len(angles))