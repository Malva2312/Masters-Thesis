import torch

import torch.fft

class FFTFeatureExtractor:
    """
    Extracts FFT-based features from 2D images or batches of 2D images.
    """

    def __init__(self):
        pass

    def __call__(self, images: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            images (torch.Tensor): 2D image (1, H, W) or batch (N, H, W), float32 or float64.
            mask (torch.Tensor, optional): Binary mask (same shape as images) to focus on nodule region.

        Returns:
            dict: {
                'fft_magnitude': torch.Tensor,
                'fft_phase': torch.Tensor,
                'fft_mean': float,
                'fft_std': float,
                ...
            }
        """
        # Ensure images are float
        images = images.float()
        if images.dim() == 2:
            images = images.unsqueeze(0)  # (1, H, W)
        N, H, W = images.shape

        if mask is not None:
            mask = mask.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            images = images * mask

        # FFT
        fft = torch.fft.fft2(images)
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.abs(fft_shifted)
        phase = torch.angle(fft_shifted)

        # Features
        features = {
            'fft_magnitude': magnitude,
            'fft_phase': phase,
            #'fft_mean': magnitude.mean(dim=(-2, -1)).cpu().numpy(),
            #'fft_std': magnitude.std(dim=(-2, -1)).cpu().numpy(),
            #'fft_max': magnitude.amax(dim=(-2, -1)).cpu().numpy(),
            #'fft_min': magnitude.amin(dim=(-2, -1)).cpu().numpy(),
        }
        return features