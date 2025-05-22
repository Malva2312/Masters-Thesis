import torch

import numpy as np
from scipy.ndimage import binary_erosion

class AreaPerimeter:
    def __call__(self, masks: torch.Tensor) -> dict:
        """
        Compute area and perimeter from a batch of binary masks.
        Args:
            masks (torch.Tensor): Binary masks of shape (N, ...), where ... can be (H, W), (D, H, W), or (C, H, W).
        Returns:
            dict: {'area': torch.Tensor, 'perimeter': torch.Tensor}, each of shape (N,)
        """
        assert masks.dim() >= 3, "Masks must be at least 3D (N, ...)"
        device = masks.device
        masks = masks.bool()
        batch_size = masks.size(0)
        areas = masks.view(batch_size, -1).sum(dim=1)

        perimeters = []
        for mask in masks:
            mask_np = mask.cpu().numpy()
            eroded = binary_erosion(mask_np)
            perimeter = np.logical_xor(mask_np, eroded).sum()
            perimeters.append(perimeter)
        perimeters = torch.tensor(perimeters, dtype=areas.dtype).to(device)
        return {"area": areas, "perimeter": perimeters}