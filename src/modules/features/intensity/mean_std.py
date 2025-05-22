import torch
class MeanStd:
    def __call__(self, imgs: torch.Tensor, masks: torch.Tensor = None) -> dict:
        """
        imgs: torch.Tensor of shape (B, ...) where B is batch size
        masks: torch.Tensor of shape (B, ...) or None
        Returns dict with keys 'mean' and 'std', each a torch.Tensor of shape (B,)
        """
        means = []
        stds = []
        batch_size = imgs.shape[0]
        for i in range(batch_size):
            img = imgs[i]
            mask = masks[i] if masks is not None else None
            if mask is not None:
                masked = img[mask.bool()]
            else:
                masked = img.flatten()
            means.append(masked.mean())
            stds.append(masked.std())
        return {"mean": torch.stack(means), "std": torch.stack(stds)}