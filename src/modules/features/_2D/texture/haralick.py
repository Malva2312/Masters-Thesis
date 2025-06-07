import torch
import numpy as np
import mahotas

class Haralick:
    def __init__(self, compute_mean=True):
        self.compute_mean = compute_mean

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

            feats = mahotas.features.haralick(img_uint8)
            if self.compute_mean:
                feats = feats.mean(axis=0)
            features_list.append(torch.from_numpy(feats).float())

        haralick_features = torch.stack(features_list)
        return {"haralick": haralick_features}