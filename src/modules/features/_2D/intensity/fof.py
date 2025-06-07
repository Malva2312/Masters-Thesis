import torch
import numpy as np
from radiomics.firstorder import RadiomicsFirstOrder
from typing import Dict, Any
import SimpleITK as sitk

class FirstOrderFeatures2D:
    def __init__(self):
        # List of desired feature keys as they appear in PyRadiomics output
        self.selected_features = [
            'Energy',
            'TotalEnergy',
            'Entropy',
            'Minimum',
            '10Percentile',
            '90Percentile',
            'Maximum',
            'Mean',
            'Median',
            'InterquartileRange',
            'Range',
            'MeanAbsoluteDeviation',
            'RobustMeanAbsoluteDeviation',
            'RootMeanSquared',
            'Skewness',
            'Kurtosis',
            'Variance',
            'Uniformity'
        ]

    def extract(self, images: torch.Tensor, masks: torch.Tensor) -> Dict[str, Any]:
        if images.dim() == 2:
            return self._extract_single(images, masks)
        elif images.dim() == 3:
            features_dict = {}
            for idx, (img, msk) in enumerate(zip(images, masks)):
                single_features = self._extract_single(img, msk)
                for k, v in single_features.items():
                    features_dict[f"{k}"] = v
            return features_dict
        else:
            raise ValueError("Input images and masks must be 2D or 3D tensors.")

    def _extract_single(self, image: torch.Tensor, mask: torch.Tensor) -> Dict[str, Any]:
        img_np = image.cpu().numpy().astype(np.float32)
        mask_np = mask.cpu().numpy().astype(np.uint8)

        # Check if mask is empty to avoid invalid calculations
        if np.count_nonzero(mask_np) == 0:
            return {k: float('nan') for k in self.selected_features}

        img_sitk = sitk.GetImageFromArray(img_np)
        mask_sitk = sitk.GetImageFromArray(mask_np)

        firstorder_extractor = RadiomicsFirstOrder(img_sitk, mask_sitk)
        for feature in self.selected_features:
            firstorder_extractor.enableFeatureByName(feature)
        results = firstorder_extractor.execute()

        # Replace NaN values in results with 0
        results = {k: (0 if np.isnan(v) else v) for k, v in results.items()}

        # Filter only the selected features
        features = {k: v for k, v in results.items() if k in self.selected_features}
        return features