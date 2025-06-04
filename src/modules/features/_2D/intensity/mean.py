import torch

class Mean:
    """
    Computes the mean intensity of nodules in 2D images using a provided mask.
    """

    def __init__(self):
        self.feature_name = "mean"

    def __call__(self, images: torch.Tensor, masks: torch.Tensor) -> dict:
        """
        Args:
            images (torch.Tensor): 2D image tensor of shape (H, W) or batch (N, H, W)
            masks (torch.Tensor): 2D mask tensor of shape (H, W) or batch (N, H, W), same shape as images

        Returns:
            dict: {'mean': float or list of floats}
        """
        if images.dim() == 2:
            # Single image
            masked_pixels = images[masks.bool()]
            mean_val = masked_pixels.float().mean().item() if masked_pixels.numel() > 0 else 0.0
            return {self.feature_name: mean_val}
        elif images.dim() == 3:
            # Batch of images
            means = []
            for img, msk in zip(images, masks):
                masked_pixels = img[msk.bool()]
                mean_val = masked_pixels.float().mean().item() if masked_pixels.numel() > 0 else 0.0
                means.append(mean_val)
            return {self.feature_name: means}
        else:
            raise ValueError("Input images and masks must be 2D or 3D tensors with matching shapes.")