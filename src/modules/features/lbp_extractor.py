import numpy as np
from skimage import feature

import torch

class LBPExtractor:
    def __init__(self, radius=1, n_points=8):
        self.radius = radius
        self.n_points = n_points

    def extract_lbp(self, image):
        if hasattr(image, "detach"):  # Check if image is a torch tensor
            image = image.detach().cpu().numpy()
        lbp = feature.local_binary_pattern(image, self.n_points, self.radius, method="uniform")
        return lbp

    def extract_features(self, images):
        features = []
        for image in images:
            lbp_features = self.extract_lbp(image)
            features.append(lbp_features)
        return np.array(features)



class LocalBinaryPattern:
    def __init__(self, P: int = 8, R: int = 1, method: str = 'uniform'):
        """
        Initialize the LBP transformer.
        
        Args:
            P (int): Number of circularly symmetric neighbor set points (quantization of the angular space).
            R (int): Radius of circle (spatial resolution of the operator).
            method (str): Method to determine the pattern. Options: 'default', 'ror', 'uniform', 'var'.
        """
        self.P = P
        self.R = R
        self.method = method

    def _apply_lbp_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies LBP to a single grayscale image.
        
        Args:
            image (np.ndarray): Grayscale image.
        
        Returns:
            np.ndarray: LBP-transformed image.
        """
        return feature.local_binary_pattern(image, self.P, self.R, self.method)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Applies LBP to a single image or batch of images.
        
        Args:
            images (torch.Tensor): Tensor of shape (H, W), (1, H, W), or (B, H, W)/(B, 1, H, W).
        
        Returns:
            torch.Tensor: Tensor of LBP images, same batch shape as input.
        """
        # Handle input shape
        original_shape = images.shape
        if len(original_shape) == 2:
            images = images.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif len(original_shape) == 3:  # (B, H, W) or (1, H, W)
            images = images.unsqueeze(1)  # (B, 1, H, W)
        elif len(original_shape) == 4 and images.shape[1] != 1:
            raise ValueError("LBP expects grayscale images with 1 channel.")

        lbp_results = []
        for img in images:
            img_np = img[0].cpu().numpy()  # Take the single channel
            lbp_np = self._apply_lbp_to_image(img_np)
            lbp_tensor = torch.from_numpy(lbp_np).float()
            lbp_results.append(lbp_tensor)

        output = torch.stack(lbp_results)  # (B, H, W)

        # Reshape to match input shape
        if len(original_shape) == 2:
            return output[0]
        elif len(original_shape) == 3:
            return output
        elif len(original_shape) == 4:
            return output.unsqueeze(1)  # (B, 1, H, W)
