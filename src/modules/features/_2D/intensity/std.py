import torch

class StandardDeviation:
    """
    Computes the standard deviation of pixel intensities within the nodule mask
    for a single 2D image or a batch of 2D images.
    """

    def __init__(self):
        pass

    def __call__(self, images: torch.Tensor, masks: torch.Tensor) -> dict:
        """
        Args:
            images (torch.Tensor): 2D image tensor of shape (H, W) or batch of images (N, H, W)
            masks (torch.Tensor): Corresponding mask tensor of shape (H, W) or (N, H, W)
        Returns:
            dict: {'std_intensity': std_value or tensor of std values}
        """
        # Ensure images and masks are float tensors
        images = images.float()
        masks = masks.float()

        # If single image, add batch dimension
        if images.dim() == 2:
            images = images.unsqueeze(0)
            masks = masks.unsqueeze(0)

        # Avoid division by zero
        eps = 1e-8

        # Compute masked mean
        masked_pixels = images * masks
        num_pixels = masks.sum(dim=(1, 2)) + eps
        mean = masked_pixels.sum(dim=(1, 2)) / num_pixels

        # Compute masked std
        mean = mean.view(-1, 1, 1)
        variance = (((images - mean) * masks) ** 2).sum(dim=(1, 2)) / num_pixels
        std = torch.sqrt(variance)

        # If input was a single image, return scalar
        if std.numel() == 1:
            std_value = std.item()
        else:
            std_value = std

        return {'std': std_value}