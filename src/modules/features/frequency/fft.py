import torch

class FastFourierTransform:
    def __init__(self, return_magnitude: bool = True, return_phase: bool = False):
        self.return_magnitude = return_magnitude
        self.return_phase = return_phase

    def _apply_fft_to_image(self, image: torch.Tensor) -> dict:
        fft_result = torch.fft.fft2(image)
        fft_result = torch.fft.fftshift(fft_result)
        result = {}
        if self.return_magnitude:
            magnitude = torch.abs(fft_result)
            result['magnitude'] = magnitude
        if self.return_phase:
            phase = torch.angle(fft_result)
            result['phase'] = phase
        return result

    def __call__(self, images: torch.Tensor, masks: torch.Tensor = None) -> dict:
        original_shape = images.shape
        if len(original_shape) == 2:
            images = images.unsqueeze(0).unsqueeze(0)
            if masks is not None:
                masks = masks.unsqueeze(0).unsqueeze(0)
        elif len(original_shape) == 3:
            images = images.unsqueeze(1)
            if masks is not None:
                masks = masks.unsqueeze(1)
        elif len(original_shape) == 4 and images.shape[1] != 1:
            raise ValueError("FastFourierTransform expects grayscale images with 1 channel.")
        if masks is not None and images.shape != masks.shape:
            raise ValueError("Masks must have the same shape as images.")

        batch_results = {k: [] for k in (['magnitude'] if self.return_magnitude else []) + (['phase'] if self.return_phase else [])}
        for idx, img in enumerate(images):
            img_tensor = img[0]
            if masks is not None:
                mask_tensor = masks[idx][0]
                img_tensor = img_tensor * mask_tensor
            fft_dict = self._apply_fft_to_image(img_tensor)
            for k, v in fft_dict.items():
                batch_results[k].append(v.float())

        for k in batch_results:
            batch_results[k] = torch.stack(batch_results[k])

        # Reshape to match input shape
        if len(original_shape) == 2:
            for k in batch_results:
                batch_results[k] = batch_results[k][0]
        elif len(original_shape) == 3:
            pass  # (B, H, W)
        elif len(original_shape) == 4:
            for k in batch_results:
                batch_results[k] = batch_results[k].unsqueeze(1)  # (B, 1, H, W)

        return batch_results
