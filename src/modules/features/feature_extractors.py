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
    "mean" : MeanStd,
    "std" : MeanStd,  # Alias for consistency
    "percentiles": Percentiles,

    # Shape features
    "area": AreaPerimeter,
    "perimeter": AreaPerimeter,  # Alias for consistency
    "area_perimeter": AreaPerimeter,  # Alias for consistency
    "eccentricity": EccentricitySolidity,
    "solidity": EccentricitySolidity, # Alias for consistency

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
        
        self.feature_dict = {}

    def __call__(self, images: torch.Tensor, masks: torch.Tensor = None) -> dict:
        """
        Extract features from a batch of images (and optional masks).
        Returns a dict: {feature_name: torch.Tensor}
        Avoids redundant calls for extractors with multiple aliases.
        """
        batch_size = images.shape[0]
        features_dict = {}

        # Map extractor class to all names/aliases used
        class_to_names = {}
        for name, extractor in zip(self.extractor_names, self.extractors):
            cls = type(extractor)
            if cls not in class_to_names:
                class_to_names[cls] = []
            class_to_names[cls].append(name)

        # Only call each extractor once, then assign results to all aliases
        called_extractors = {}
        for extractor, names in zip(self.extractors, self.extractor_names):
            cls = type(extractor)
            if cls in called_extractors:
                continue  # Already called this extractor
            try:
                feat = extractor(images, masks)
            except TypeError:
                feat = extractor(images)
            called_extractors[cls] = feat

            # Assign features to all aliases for this extractor
            for name in class_to_names[cls]:
                if isinstance(feat, torch.Tensor):
                    features_dict[name] = feat.view(batch_size, -1)
                elif isinstance(feat, dict):
                    for key, value in feat.items():
                        if isinstance(value, torch.Tensor):
                            features_dict[f"{name}_{key}"] = value.view(batch_size, -1)
                        else:
                            raise ValueError(f"Unexpected type in dict: {type(value)}")
                else:
                    raise ValueError(f"Unexpected feature type: {type(feat)}")

        self.feature_dict = features_dict
        return features_dict

    def to_vector(self) -> torch.Tensor:
        """
        Convert the extracted features to a single vector.
        """
        if not self.feature_dict:
            raise ValueError("No features extracted. Call the extractor first.")
        
        # Concatenate all feature tensors into a single vector
        feature_vectors = [feat.view(feat.size(0), -1) for key, feat in self.feature_dict.items() if key in self.extractor_names]
        return torch.cat(feature_vectors, dim=1)