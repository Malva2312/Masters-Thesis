import torch
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte

class GLCM:
    def __init__(self, distances=[1], angles=[0], levels=256, props=None):
        self.distances = distances
        self.angles = angles
        self.levels = levels
        if props is None:
            self.props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        else:
            self.props = props

    def __call__(self, images: torch.Tensor, masks: torch.Tensor = None):
        """
        Args:
            images (torch.Tensor): (H, W) or (B, H, W) grayscale images
            masks (torch.Tensor, optional): (H, W) or (B, H, W) binary masks

        Returns:
            dict: {'glcm_features': torch.Tensor of shape (B, num_props)}
        """
        if images.dim() == 2:
            images = images.unsqueeze(0)
            if masks is not None:
                masks = masks.unsqueeze(0)
        assert images.dim() == 3, "Images should be of shape (B, H, W) or (H, W)"
        if masks is not None:
            assert masks.shape == images.shape

        glcm_features = []
        for i in range(images.shape[0]):
            img = images[i].cpu().numpy()
            img_u8 = img_as_ubyte((img - img.min()) / (img.ptp() + 1e-8))

            if masks is not None:
                mask = masks[i].cpu().numpy().astype(bool)
                img_u8 = img_u8 * mask

            glcm = graycomatrix(
                img_u8,
                distances=self.distances,
                angles=self.angles,
                levels=self.levels,
                symmetric=True,
                normed=True
            )
            feature_vec = []
            for prop in self.props:
                # Take the mean over all distances and angles
                vals = graycoprops(glcm, prop)
                feature_vec.append(np.mean(vals))
            glcm_features.append(torch.tensor(feature_vec, dtype=torch.float32))
        result = torch.stack(glcm_features, dim=0)
        return {"glcm": result}
        
