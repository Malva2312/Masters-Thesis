import torch
import numpy as np
from skimage import feature

class LocalBinaryPattern:
    def __init__(self, P: int = 8, R: int = 1, method: str = 'uniform'):
        self.P = P
        self.R = R
        self.method = method

    def _apply_lbp_to_image(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        if mask is not None:
            image = image * (mask > 0)
        return feature.local_binary_pattern(image, self.P, self.R, self.method)

    def __call__(self, images: torch.Tensor, masks: np.ndarray = None) -> torch.Tensor:
        original_shape = images.shape
        device = images.device  # Save the original device

        if len(original_shape) == 2:
            images = images.unsqueeze(0).unsqueeze(0)
        elif len(original_shape) == 3:
            images = images.unsqueeze(1)
        elif len(original_shape) == 4 and images.shape[1] != 1:
            raise ValueError("LBP expects grayscale images with 1 channel.")

        lbp_results = []
        for i in range(images.shape[0]):
            img = images[i]
            mask = masks[i] if masks is not None else None
            img_np = img[0].cpu().numpy()  # Take the single channel
            lbp_np = self._apply_lbp_to_image(img_np, mask)
            lbp_tensor = torch.from_numpy(lbp_np).float().to(device)
            lbp_results.append(lbp_tensor)

        output = torch.stack(lbp_results)  # (B, H, W)

        # Reshape to match input shape
        if len(original_shape) == 2:
            result = output[0]
        elif len(original_shape) == 3:
            result = output
        elif len(original_shape) == 4:
            result = output.unsqueeze(1)  # (B, 1, H, W)
        else:
            result = output

        return result
