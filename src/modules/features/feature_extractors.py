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
        self.extractors = []
        self.extractor_names = []
        
        for extractor_cfg in config.get("extractors", []):
            name = extractor_cfg["name"]
            params = extractor_cfg.get("params", {})
            extractor_class = FEATURE_EXTRACTOR_CLASSES.get(name)
            if extractor_class is None:
                raise ValueError(f"Unknown extractor name: {name}")
            self.extractors.append(extractor_class(**params))
            self.extractor_names.append(name)

    def __call__(self, images: torch.Tensor, masks: torch.Tensor = None) -> dict:
        """
        Extract features from a batch of images (and optional masks).
        Returns a dict: {feature_name: torch.Tensor}
        """
        batch_size = images.shape[0]
        features_dict = {}

        for name, extractor in zip(self.extractor_names, self.extractors):
            try:
                feat = extractor(images, masks)
            except TypeError:
                feat = extractor(images)
            if isinstance(feat, torch.Tensor):
                feat = feat.view(batch_size, -1)
                features_dict[name] = feat
            elif isinstance(feat, dict):
                for key, value in feat.items():
                    if isinstance(value, torch.Tensor):
                        value = value.view(batch_size, -1)
                        features_dict[f"{name}_{key}"] = value
                    else:
                        raise ValueError(f"Unexpected type in dict: {type(value)}")
            else:
                raise ValueError(f"Unexpected feature type: {type(feat)}")

        return features_dict