import torch

class EccentricitySolidity:
    def __call__(self, masks: torch.Tensor) -> dict:
        """
        Compute eccentricity and solidity from a batch of binary masks.
        Args:
            masks (torch.Tensor): Binary masks of shape (B, ...), where ... can be (H, W), (D, H, W), etc.
        Returns:
            dict: {'eccentricity': torch.Tensor, 'solidity': torch.Tensor}
        """
        assert masks.dim() >= 3, "Mask must be at least 3D (B, ...)"
        masks = masks.bool()
        import numpy as np
        from skimage.measure import regionprops, label

        batch_size = masks.shape[0]
        eccentricities = []
        solidities = []
        for i in range(batch_size):
            mask_np = masks[i].cpu().numpy().astype(np.uint8)
            labeled = label(mask_np)
            props = regionprops(labeled)
            if len(props) == 0:
                eccentricities.append(float('nan'))
                solidities.append(float('nan'))
            else:
                prop = props[0]
                # regionprops.eccentricity and solidity are only defined for 2D
                if mask_np.ndim == 2:
                    eccentricities.append(float(prop.eccentricity))
                    solidities.append(float(prop.solidity))
                else:
                    # For 2.5D/3D, set as nan (not supported by skimage)
                    eccentricities.append(float('nan'))
                    solidities.append(float('nan'))
        return {
            "eccentricity": torch.tensor(eccentricities, dtype=torch.float32, device=masks.device),
            "solidity": torch.tensor(solidities, dtype=torch.float32, device=masks.device)
        }