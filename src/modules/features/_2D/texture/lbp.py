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
        if images.dim() == 2:
            images = images.unsqueeze(0)
            if masks is not None:
                masks = masks.unsqueeze(0)

        assert images.dim() == 3, "Images should be of shape (B, H, W) or (H, W)"
        if masks is not None:
            assert masks.shape == images.shape

        img0 = images[0].detach().cpu().numpy()
        img0_u8 = img_as_ubyte((img0 - img0.min()) / (img0.ptp() + 1e-8))
        lbp0 = local_binary_pattern(img0_u8, P=self.n_points, R=self.radius, method=self.method)
        if self.method == 'uniform':
            n_bins = self.n_points + 2
        else:
            n_bins = int(lbp0.max() + 1)

        lbp_features = torch.zeros((images.shape[0], n_bins), dtype=torch.float32)

        for i in range(images.shape[0]):
            img = images[i].detach().cpu().numpy()
            img_u8 = img_as_ubyte((img - img.min()) / (img.ptp() + 1e-8))

            lbp = local_binary_pattern(img_u8, P=self.n_points, R=self.radius, method=self.method)
            if masks is not None:
                mask = masks[i].detach().cpu().numpy().astype(bool)
                lbp = lbp[mask]

            hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
            lbp_features[i] = torch.from_numpy(hist).float()

        return {"lbp": lbp_features}
