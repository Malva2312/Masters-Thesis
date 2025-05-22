import torch

import torch.nn.functional as F

class HistogramOfOrientedGradients:
    def __init__(self, cell_size=8, block_size=2, nbins=9):
        self.cell_size = cell_size
        self.block_size = block_size
        self.nbins = nbins

    def __call__(self, img: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute HOG features for a single image or a batch of images.
        Args:
            img (torch.Tensor): Image tensor of shape (H, W), (1, H, W), (C, H, W) with C=1, or (N, H, W).
            mask (torch.Tensor, optional): Binary mask tensor of shape (H, W) or (N, H, W). If provided, only masked pixels contribute.
        Returns:
            torch.Tensor: HOG feature vector or batch of feature vectors.
        """
        # Handle batch input
        if img.dim() == 4 and img.size(1) == 1:
            # (N, 1, H, W) -> (N, H, W)
            img = img.squeeze(1)
        if img.dim() == 3 and img.size(0) > 1:
            # (N, H, W)
            imgs = img
            if mask is not None:
                assert mask.shape == imgs.shape, "Mask must have the same shape as the image batch"
                return torch.stack([self.__call__(im, m) for im, m in zip(imgs, mask)])
            else:
                return torch.stack([self.__call__(im) for im in imgs])
        # Now img is (H, W) or (1, H, W) or (C, H, W) with C=1
        if img.dim() == 3:
            img = img.squeeze(0)
        assert img.dim() == 2, "Input must be a grayscale image tensor of shape (H, W)"

        if mask is not None:
            assert mask.shape == img.shape, "Mask must have the same shape as the image"
            mask = mask.bool()

        # Compute gradients
        img_ = img.unsqueeze(0)  # (1, H, W)
        gx = F.pad(img_, (1, 1, 0, 0), mode='replicate')[:, :, 2:] - F.pad(img_, (1, 1, 0, 0), mode='replicate')[:, :, :-2]
        gy = F.pad(img_, (0, 0, 1, 1), mode='replicate')[:, 2:, :] - F.pad(img_, (0, 0, 1, 1), mode='replicate')[:, :-2, :]
        gx = gx.squeeze(0)
        gy = gy.squeeze(0)

        magnitude = torch.sqrt(gx ** 2 + gy ** 2)
        orientation = torch.atan2(gy, gx) * (180.0 / torch.pi) % 180  # [0, 180)

        # Quantize orientations
        bin_width = 180 / self.nbins
        bins = torch.floor(orientation / bin_width).long()
        bins = torch.clamp(bins, max=self.nbins - 1)

        # Divide into cells
        H, W = img.shape
        n_cells_y = H // self.cell_size
        n_cells_x = W // self.cell_size

        hog = torch.zeros((n_cells_y, n_cells_x, self.nbins), dtype=img.dtype, device=img.device)

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                y0, y1 = i * self.cell_size, (i + 1) * self.cell_size
                x0, x1 = j * self.cell_size, (j + 1) * self.cell_size
                cell_mag = magnitude[y0:y1, x0:x1]
                cell_bin = bins[y0:y1, x0:x1]
                if mask is not None:
                    cell_mask = mask[y0:y1, x0:x1]
                for b in range(self.nbins):
                    if mask is not None:
                        hog[i, j, b] = (cell_mag[(cell_bin == b) & cell_mask]).sum()
                    else:
                        hog[i, j, b] = (cell_mag[cell_bin == b]).sum()

        # Block normalization
        eps = 1e-6
        blocks = []
        for i in range(n_cells_y - self.block_size + 1):
            for j in range(n_cells_x - self.block_size + 1):
                block = hog[i:i+self.block_size, j:j+self.block_size, :].reshape(-1)
                block = block / (block.norm(p=2) + eps)
                blocks.append(block)
        hog_vec = torch.cat(blocks)
        return hog_vec