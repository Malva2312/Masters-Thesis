import torch

class Entropy:
    def __call__(self, imgs: torch.Tensor, mask: torch.Tensor = None, num_bins: int = 256) -> torch.Tensor:
        """
        imgs: torch.Tensor of shape (N, ...) where N is the batch or slice dimension.
        mask: Optional torch.Tensor of same shape as imgs, or broadcastable.
        Returns: torch.Tensor of entropies, shape (N,)
        """
        # Flatten all but first dimension
        N = imgs.shape[0]
        entropies = []
        for i in range(N):
            img = imgs[i]
            msk = mask[i] if mask is not None else None
            if msk is not None:
                values = img[msk.bool()].flatten()
            else:
                values = img.flatten()
            if values.numel() == 0:
                entropies.append(torch.tensor(float('nan'), device=imgs.device))
                continue
            hist = torch.histc(values, bins=num_bins, min=float(values.min()), max=float(values.max()))
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            entropy = -(prob * torch.log2(prob)).sum()
            entropies.append(entropy)
        return torch.stack(entropies)