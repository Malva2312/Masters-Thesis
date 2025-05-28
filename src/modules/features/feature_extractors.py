import torch

from src.modules.features.frequency.fft import FastFourierTransform
from src.modules.features.frequency.gabor import GaborFeature
from src.modules.features.gradient.hog import HistogramOfOrientedGradients
from src.modules.features.intensity.entropy import Entropy
from src.modules.features.intensity.mean_std import MeanStd
from src.modules.features.intensity.percentiles import Percentiles
from src.modules.features.shape.area_perimeter import AreaPerimeter
from src.modules.features.shape.eccentricity_solidity import EccentricitySolidity
from src.modules.features.texture.glcm import GrayLevelCooccurrenceMatrix
#from src.modules.features.texture.glrlm import GLRLMFeature
from src.modules.features.texture.lbp import LocalBinaryPattern
import torch
#from modules.features.keypoints.brief import BRIEFFeature

FEATURE_EXTRACTOR_CLASSES = {
    # Frequency features
    "fft": FastFourierTransform,
    "gabor": GaborFeature,
    
    # Gradient features
    "hog": HistogramOfOrientedGradients,

    # Intensity features
    "entropy": Entropy,
    "mean_std": MeanStd,
    "percentiles": Percentiles,

    # Shape features
    "areaperimeter": AreaPerimeter,
    "eccentricity_solidity": EccentricitySolidity,

    # Texture features
    "glcm": GrayLevelCooccurrenceMatrix,
    "lbp": LocalBinaryPattern,

    # Keypoints features
    # "brief": BRIEFFeature,
}

class FeatureExtractorManager:
    def __init__(self, config):
        self.total_dim = 0
        self.extractors = []
        
        for extractor_cfg in config.get("extractors", []):
            name = extractor_cfg["name"]
            params = extractor_cfg.get("params", {})
            extractor_class = FEATURE_EXTRACTOR_CLASSES.get(name)
            if extractor_class is None:
                raise ValueError(f"Unknown extractor name: {name}")
            self.extractors.append(extractor_class(**params))

        self.get_total_feature_dim()  # Initialize total_dim

    def __call__(self, images: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        """
        Extract features from a batch of images (and optional masks).
        Returns a 2D torch.Tensor: (batch_size, total_features)
        """
        batch_size = images.shape[0]
        feature_list = []

        for extractor in self.extractors:
            try:
                feat = extractor(images, masks)
            except TypeError:
                feat = extractor(images)
            if isinstance(feat, torch.Tensor):
                # Flatten all but batch dimension
                feat = feat.view(batch_size, -1)
                feature_list.append(feat)
            elif isinstance(feat, dict):
                # Concatenate all tensors in the dict along feature dimension
                for value in feat.values():
                    if isinstance(value, torch.Tensor):
                        value = value.view(batch_size, -1)
                        feature_list.append(value)
                    else:
                        raise ValueError(f"Unexpected type in dict: {type(value)}")
            else:
                raise ValueError(f"Unexpected feature type: {type(feat)}")

        if feature_list:
            features = torch.cat(feature_list, dim=1)  # (batch_size, total_features)
            return features
        else:
            return torch.empty((batch_size, 0))

    def get_total_feature_dim(self, sample_image_shape=(1, 32, 32)):
        """
        Returns the total feature dimension for the configured extractors,
        given a sample image shape (default: (1, 32, 32)).
        Also updates self.total_dim.
        """
        # Create a dummy batch of size 1
        dummy_image = torch.zeros((1, *sample_image_shape))

        feature_list = []
        for extractor in self.extractors:
            try:
                feat = extractor(dummy_image)
            except TypeError:
                feat = extractor(dummy_image)
            if isinstance(feat, torch.Tensor):
                feat = feat.view(1, -1)
                feature_list.append(feat)
            elif isinstance(feat, dict):
                for value in feat.values():
                    if isinstance(value, torch.Tensor):
                        value = value.view(1, -1)
                        feature_list.append(value)
                    else:
                        raise ValueError(f"Unexpected type in dict: {type(value)}")
            else:
                raise ValueError(f"Unexpected feature type: {type(feat)}")

        if feature_list:
            total_dim = sum(f.shape[1] for f in feature_list)
            self.total_dim = total_dim
            return total_dim
        else:
            self.total_dim = 0
            return 0
