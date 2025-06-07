import torch
import numpy as np
import mahotas

class Zernike:
    def __init__(self, radius=None, degree=8):
        """
        Args:
            radius (int, optional): Radius for Zernike moments. If None, uses half the min(H, W).
            degree (int): Maximum degree of Zernike moments.
        """
        self.radius = radius
        self.degree = degree

    def __call__(self, images: torch.Tensor, masks: torch.Tensor = None):
        if images.dim() == 2:
            images = images.unsqueeze(0)
            if masks is not None:
                masks = masks.unsqueeze(0)

        assert images.dim() == 3, "Images should be of shape (B, H, W) or (H, W)"
        if masks is not None:
            assert masks.shape == images.shape

        features_list = []
        for i in range(images.shape[0]):
            img = images[i]
            img_uint8 = ((img + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
            if masks is not None:
                mask = masks[i].cpu().numpy().astype(bool)
                img_uint8 = img_uint8 * mask

            # Use the mask as the binary region for Zernike, or threshold the image if no mask
            if masks is not None:
                region = mask
            else:
                region = img_uint8 > 0

            h, w = img_uint8.shape
            radius = self.radius if self.radius is not None else min(h, w) // 2

            feats = mahotas.features.zernike_moments(region.astype(np.uint8), radius, degree=self.degree)
            features_list.append(torch.from_numpy(np.array(feats)).float())

        zernike_features = torch.stack(features_list)
        return {"zernike": zernike_features}