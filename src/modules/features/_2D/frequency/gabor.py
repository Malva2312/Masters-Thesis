import torch
import numpy as np
from typing import Dict, Union

import torch.nn.functional as F

class Gabor:
    def __init__(self, frequencies=[0.1, 0.2, 0.3], thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4], kernel_size=15):
        self.frequencies = frequencies
        self.thetas = thetas
        self.kernel_size = kernel_size
        self.kernels = self._create_gabor_kernels()

    def _create_gabor_kernels(self):
        kernels = []
        for freq in self.frequencies:
            for theta in self.thetas:
                kernel = self._gabor_kernel(self.kernel_size, freq, theta)
                kernels.append((freq, theta, kernel))
        return kernels

    def _gabor_kernel(self, size, frequency, theta, sigma_x=None, sigma_y=None):
        if sigma_x is None:
            sigma_x = size / 6.0
        if sigma_y is None:
            sigma_y = size / 6.0
        xmax = size // 2
        ymax = size // 2
        xmin = -xmax
        ymin = -ymax
        y, x = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1))
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        gb = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * np.cos(2 * np.pi * frequency * x_theta)
        return torch.tensor(gb, dtype=torch.float32)

    def __call__(self, images: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        images: (B, 1, H, W) or (1, H, W) or (H, W)
        mask: (B, 1, H, W) or (1, H, W) or (H, W)
        Returns: dict with a single key 'gabor' and value as a tensor of shape (B, num_kernels)
        """
        if images.dim() == 2:
            images = images.unsqueeze(0).unsqueeze(0)
        elif images.dim() == 3:
            images = images.unsqueeze(1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        feature_list = []
        for freq, theta, kernel in self.kernels:
            kernel = kernel.to(images.device)
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # (out_channels, in_channels, kH, kW)
            filtered = F.conv2d(images, kernel, padding=self.kernel_size//2)
            masked = filtered * mask
            mean_val = masked.sum(dim=[2,3]) / (mask.sum(dim=[2,3]) + 1e-8)
            mean_val = mean_val.view(mean_val.size(0), 1)  # Ensure shape (B, 1)
            feature_list.append(mean_val)  # (B, 1)
        features_tensor = torch.cat(feature_list, dim=1)  # (B, num_kernels)
        return {'gabor': features_tensor}