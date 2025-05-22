import torch

class Percentiles:
    def __init__(self, percentiles=(25, 50, 75)):
        self.percentiles = percentiles

    def __call__(self, imgs: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        """
        imgs: torch.Tensor of shape (B, ...) where B is batch size
        masks: torch.Tensor of shape (B, ...) or None
        Returns: torch.Tensor of shape (B, num_percentiles)
        """
        batch_size = imgs.shape[0]
        num_percentiles = len(self.percentiles)
        results = torch.empty((batch_size, num_percentiles), dtype=imgs.dtype, device=imgs.device)
        for i in range(batch_size):
            img = imgs[i]
            mask = masks[i] if masks is not None else None
            if mask is not None:
                values = img[mask.bool()].flatten()
            else:
                values = img.flatten()
            percentiles_tensor = torch.tensor(
                [torch.quantile(values, p / 100.0) for p in self.percentiles],
                dtype=imgs.dtype,
                device=imgs.device
            )
            results[i] = percentiles_tensor
        return results