import torch
from typing import Dict, Union

import torch.nn.functional as F

class Entropy:
    """
    Computes the entropy of a masked region in a 2D image or a batch of 2D images.
    """

    def __init__(self, num_bins: int = 256):
        self.num_bins = num_bins

    def __call__(
        self, 
        images: torch.Tensor, 
        masks: torch.Tensor
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Args:
            images (torch.Tensor): 2D image (H, W) or batch (N, H, W)
            masks (torch.Tensor): 2D mask (H, W) or batch (N, H, W), same shape as images

        Returns:
            Dict[str, Union[float, torch.Tensor]]: {'entropy': value or tensor}
        """
        if images.dim() == 2:
            images = images.unsqueeze(0)
            masks = masks.unsqueeze(0)

        entropies = []
        for img, msk in zip(images, masks):
            masked_pixels = img[msk > 0]
            if masked_pixels.numel() == 0:
                entropies.append(torch.tensor(0.0, device=img.device))
                continue
            # Normalize to [0, 1]
            min_val = masked_pixels.min()
            max_val = masked_pixels.max()
            if max_val > min_val:
                norm_pixels = (masked_pixels - min_val) / (max_val - min_val)
            else:
                norm_pixels = masked_pixels * 0  # all zeros
            # Histogram
            hist = torch.histc(norm_pixels, bins=self.num_bins, min=0.0, max=1.0)
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            entropy = -torch.sum(prob * torch.log2(prob))
            entropies.append(entropy)

        entropies = torch.stack(entropies)
        if entropies.shape[0] == 1:
            return {'entropy': entropies[0].item()}
        else:
            return {'entropy': entropies}

# Example usage:
# entropy_extractor = Entropy()
# result = entropy_extractor(images, masks)