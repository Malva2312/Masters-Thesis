import torch
from src.modules.features._2D.frequency.fft import FFTFeatureExtractor
from src.modules.features._2D.frequency.gabor import Gabor
from src.modules.features._2D.gradient.hog import HOGFeatureExtractor
from src.modules.features._2D.intensity.entropy import Entropy
from src.modules.features._2D.intensity.mean import Mean
from src.modules.features._2D.intensity.std import StandardDeviation
from src.modules.features._2D.shape.shape import ShapeFeatures2D
from src.modules.features._2D.texture.glcm import GLCM
from src.modules.features._2D.texture.lbp import LBP

class FeatureExtractorManager:
    """
    Manages and applies all feature extractors to a batch of images and masks.
    Returns a dictionary with feature names as keys and torch tensors as values.
    """

    def __init__(self):
        self.fft_extractor = FFTFeatureExtractor()
        #self.gabor_extractor = Gabor()
        self.hog_extractor = HOGFeatureExtractor()
        self.entropy_extractor = Entropy()
        self.mean_extractor = Mean()
        self.std_extractor = StandardDeviation()
        self.shape_extractor = ShapeFeatures2D()
        self.glcm_extractor = GLCM()
        self.lbp_extractor = LBP()

        # Shape feature keys to extract
        self.shape_keys = [
            'MeshSurface',
            'Perimeter',
            'PerimeterSurfaceRatio',
            'Sphericity',
            'SphericalDisproportion',
            'Maximum2DDiameter'
        ]

    def __call__(self, images: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            images: (B, H, W) torch.Tensor
            masks: (B, H, W) torch.Tensor

        Returns:
            dict: {feature_name: torch.Tensor}
        """
        features = {}

        # FFT
        fft_feats = self.fft_extractor(images, masks)
        features['fft_magnitude'] = fft_feats['fft_magnitude']
        features['fft_phase'] = fft_feats['fft_phase']

        # Gabor
        #gabor_feats = self.gabor_extractor.extract(images, masks)
        #features['gabor'] = gabor_feats['gabor']

        # HOG
        hog_feats = self.hog_extractor(images, masks)
        hog_tensor = torch.tensor(hog_feats['hog'], dtype=torch.float32)
        features['hog'] = hog_tensor

        # Entropy
        entropy_feats = self.entropy_extractor(images, masks)
        entropy_tensor = entropy_feats['entropy']
        if not isinstance(entropy_tensor, torch.Tensor):
            entropy_tensor = torch.tensor([entropy_tensor], dtype=torch.float32)
        features['entropy'] = entropy_tensor if entropy_tensor.dim() > 0 else entropy_tensor.unsqueeze(0)

        # Mean
        mean_feats = self.mean_extractor(images, masks)
        mean_tensor = mean_feats['mean']
        if isinstance(mean_tensor, list):
            mean_tensor = torch.tensor(mean_tensor, dtype=torch.float32)
        else:
            mean_tensor = torch.tensor([mean_tensor], dtype=torch.float32)
        features['mean'] = mean_tensor

        # Std
        std_feats = self.std_extractor(images, masks)
        std_tensor = std_feats['std']
        if isinstance(std_tensor, torch.Tensor):
            if std_tensor.dim() == 0:
                std_tensor = std_tensor.unsqueeze(0)
        else:
            std_tensor = torch.tensor([std_tensor], dtype=torch.float32)
        features['std'] = std_tensor

        # Shape features
        shape_feats = self.shape_extractor.extract(images, masks)
        # shape_feats may be a dict with keys like 'MeshSurface_0', ... for batch
        # We'll collect them into a tensor of shape (B, len(shape_keys))
        batch_size = images.shape[0]
        shape_tensor = torch.zeros((batch_size, len(self.shape_keys)), dtype=torch.float32)
        for i in range(batch_size):
            for j, key in enumerate(self.shape_keys):
                # Try batch key, fallback to single key
                batch_key = f"{key}_{i}"
                if batch_key in shape_feats:
                    val = shape_feats[batch_key]
                elif key in shape_feats:
                    val = shape_feats[key]
                else:
                    val = 0.0
                shape_tensor[i, j] = float(val) if val is not None else 0.0
        for idx, key in enumerate(self.shape_keys):
            features[key] = shape_tensor[:, idx]

        # GLCM
        glcm_feats = self.glcm_extractor(images, masks)
        features['glcm'] = glcm_feats['glcm']

        # LBP
        lbp_feats = self.lbp_extractor(images, masks)
        features['lbp'] = lbp_feats['lbp']

        return features