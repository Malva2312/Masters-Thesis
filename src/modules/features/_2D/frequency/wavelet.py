import torch
import numpy as np
import pywt

class Wavelet:
    def __init__(self, wavelet='db1', mode='symmetric'):
        self.wavelet = wavelet
        self.mode = mode

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
            img = images[i].detach().cpu().numpy()
            img_norm = (img - img.min()) / (img.ptp() + 1e-8)
            if masks is not None:
                mask = masks[i].detach().cpu().numpy().astype(bool)
                img_norm = img_norm * mask

            coeffs2 = pywt.dwt2(img_norm, self.wavelet, mode=self.mode)
            cA, (cH, cV, cD) = coeffs2

            # Compute statistics for each subband
            feats = []
            for subband in [cA, cH, cV, cD]:
                feats.append(subband.mean())
                feats.append(subband.std())
            features_list.append(torch.tensor(feats, dtype=torch.float32))

        wavelet_features = torch.stack(features_list)
        return {"wavelet": wavelet_features}