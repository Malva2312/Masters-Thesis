import torch
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte

class LBP:
    def __init__(self, radius=1, n_points=8, method='uniform'):
        self.radius = radius
        self.n_points = n_points
        self.method = method

    def __call__(self, images: torch.Tensor, masks: torch.Tensor = None):
        """
        Args:
            images (torch.Tensor): (H, W) or (B, H, W) grayscale images
            masks (torch.Tensor, optional): (H, W) or (B, H, W) binary masks

        Returns:
            dict: {'lbp': torch.Tensor}
        """
        if images.dim() == 2:
            images = images.unsqueeze(0)
            if masks is not None:
                masks = masks.unsqueeze(0)
        assert images.dim() == 3, "Images should be of shape (B, H, W) or (H, W)"
        if masks is not None:
            assert masks.shape == images.shape

        # Determine n_bins from the first image
        img0 = images[0].cpu().numpy()
        img0_u8 = img_as_ubyte((img0 - img0.min()) / (img0.ptp() + 1e-8))
        lbp0 = local_binary_pattern(img0_u8, P=self.n_points, R=self.radius, method=self.method)
        n_bins = int(lbp0.max() + 1)

        lbp_features = torch.zeros((images.shape[0], n_bins), dtype=torch.float32)
        for i in range(images.shape[0]):
            img = images[i].cpu().numpy()
            img_u8 = img_as_ubyte((img - img.min()) / (img.ptp() + 1e-8))

            if masks is not None:
                mask = masks[i].cpu().numpy().astype(bool)
                img_u8 = img_u8 * mask
            else:
                mask = None

            lbp = local_binary_pattern(img_u8, P=self.n_points, R=self.radius, method=self.method)
            hist, _ = np.histogram(
                lbp[mask] if mask is not None else lbp,
                bins=n_bins,
                range=(0, n_bins),
                density=True
            )
            lbp_features[i] = torch.tensor(hist, dtype=torch.float32)
        return {"lbp": lbp_features}
