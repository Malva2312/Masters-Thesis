import torch
import numpy as np
from scipy import ndimage

class GaborFeature:
    """
    Extracts Gabor features from images.
    """

    def __init__(self, frequencies=(0.1, 0.2, 0.3), thetas=(0, np.pi/4, np.pi/2)):
        self.frequencies = frequencies
        self.thetas = thetas

    def _build_gabor_kernel(self, frequency, theta, sigma_x=2.0, sigma_y=2.0):
        size = int(8 * max(sigma_x, sigma_y))
        x, y = np.meshgrid(np.arange(-size//2, size//2 + 1),
                           np.arange(-size//2, size//2 + 1))
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        gb = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * \
             np.cos(2 * np.pi * frequency * x_theta)
        return gb

    def _extract_features_from_image(self, image: np.ndarray) -> np.ndarray:
        features = []
        for freq in self.frequencies:
            for theta in self.thetas:
                kernel = self._build_gabor_kernel(freq, theta)
                filtered = ndimage.convolve(image, kernel, mode='reflect')
                features.append(filtered.mean())
                features.append(filtered.var())
        return np.array(features, dtype=np.float32)

    def __call__(self, images: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Input images, shape (B, 1, H, W), (B, H, W), (H, W), or (N, H, W).
            mask (torch.Tensor, optional): Binary mask(s) with same spatial shape as images. 
                                        Shape should broadcast to images.

        Returns:
            torch.Tensor: Extracted features.
        """
        original_device = images.device
        original_shape = images.shape
        if len(original_shape) == 2:
            images = images.unsqueeze(0).unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(original_shape) == 3:
            images = images.unsqueeze(1)
            if mask is not None and mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask is not None and mask.dim() == 3:
                mask = mask.unsqueeze(1)
        elif len(original_shape) == 4 and images.shape[1] != 1:
            raise ValueError("GaborFeatureExtractor expects grayscale images with 1 channel.")

        feature_list = []
        for idx, img in enumerate(images):
            img_np = img[0].cpu().numpy()
            if mask is not None:
                mask_np = mask[idx][0].cpu().numpy().astype(bool)
            else:
                mask_np = None

            features = []
            for freq in self.frequencies:
                for theta in self.thetas:
                    kernel = self._build_gabor_kernel(freq, theta)
                    filtered = ndimage.convolve(img_np, kernel, mode='reflect')
                    if mask_np is not None:
                        masked_filtered = filtered[mask_np]
                        features.append(masked_filtered.mean() if masked_filtered.size > 0 else 0.0)
                        features.append(masked_filtered.var() if masked_filtered.size > 0 else 0.0)
                    else:
                        features.append(filtered.mean())
                        features.append(filtered.var())
            feature_list.append(torch.from_numpy(np.array(features, dtype=np.float32)))

        output = torch.stack(feature_list)  # (B, F)

        # Reshape to match input shape
        if len(original_shape) == 2:
            return output[0].to(original_device)
        elif len(original_shape) == 3:
            return output.to(original_device)
        elif len(original_shape) == 4:
            return output.to(original_device)  # (B, F)