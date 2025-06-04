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
        self.gabor_extractor = Gabor()
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
            #'PerimeterSurfaceRatio',
            'Sphericity',
            #'SphericalDisproportion',
            #'Maximum2DDiameter'
        ]

        self.feature_dims = {
            'fft_magnitude': None,
            'fft_phase': None,
            'gabor': None,
            'hog': None,
            'entropy': None,
            'mean': None,
            'std': None,
            'glcm': None,
            'lbp': None,
        }
        for key in self.shape_keys:
            self.feature_dims[key] = None

    def __call__(self, images: torch.Tensor, masks: torch.Tensor):
        features = {}

        # FFT
        fft_feats = self.fft_extractor(images, masks)
        features['fft_magnitude'] = fft_feats['fft_magnitude']
        features['fft_phase'] = fft_feats['fft_phase']
        self.feature_dims['fft_magnitude'] = features['fft_magnitude'].shape
        self.feature_dims['fft_phase'] = features['fft_phase'].shape

        # Gabor
        gabor_feats = self.gabor_extractor(images, masks)
        gabor_tensor = gabor_feats['gabor']
        if not isinstance(gabor_tensor, torch.Tensor):
            gabor_tensor = torch.tensor(gabor_tensor, dtype=torch.float32)
        # Ensure C, H, W format
        if gabor_tensor.dim() == 2:
            gabor_tensor = gabor_tensor.unsqueeze(0)
        elif gabor_tensor.dim() == 1:
            gabor_tensor = gabor_tensor.unsqueeze(0).unsqueeze(-1)
        features['gabor'] = gabor_tensor
        self.feature_dims['gabor'] = gabor_tensor.shape

        # HOG
        hog_feats = self.hog_extractor(images, masks)
        hog_tensor = torch.tensor(hog_feats['hog'], dtype=torch.float32)
        # Ensure C, H, W format
        if hog_tensor.dim() == 2:
            hog_tensor = hog_tensor.unsqueeze(0)
        elif hog_tensor.dim() == 1:
            hog_tensor = hog_tensor.unsqueeze(0).unsqueeze(-1)
        features['hog'] = hog_tensor
        self.feature_dims['hog'] = hog_tensor.shape

        # Entropy
        entropy_feats = self.entropy_extractor(images, masks)
        entropy_tensor = entropy_feats['entropy']
        if not isinstance(entropy_tensor, torch.Tensor):
            entropy_tensor = torch.tensor([entropy_tensor], dtype=torch.float32)
        if entropy_tensor.dim() == 0:
            entropy_tensor = entropy_tensor.unsqueeze(0)
        # Ensure C, H, W format
        if entropy_tensor.dim() == 1:
            entropy_tensor = entropy_tensor.unsqueeze(-1)
        if entropy_tensor.dim() == 2:
            entropy_tensor = entropy_tensor.unsqueeze(0)
        features['entropy'] = entropy_tensor
        self.feature_dims['entropy'] = entropy_tensor.shape

        # Mean
        mean_feats = self.mean_extractor(images, masks)
        mean_tensor = mean_feats['mean']
        if isinstance(mean_tensor, list):
            mean_tensor = torch.tensor(mean_tensor, dtype=torch.float32)
        else:
            mean_tensor = torch.tensor([mean_tensor], dtype=torch.float32)
        # Ensure C, H, W format
        if mean_tensor.dim() == 1:
            mean_tensor = mean_tensor.unsqueeze(-1)
        if mean_tensor.dim() == 2:
            mean_tensor = mean_tensor.unsqueeze(0)
        features['mean'] = mean_tensor
        self.feature_dims['mean'] = mean_tensor.shape

        # Std
        std_feats = self.std_extractor(images, masks)
        std_tensor = std_feats['std']
        if isinstance(std_tensor, torch.Tensor):
            if std_tensor.dim() == 0:
                std_tensor = std_tensor.unsqueeze(0)
        else:
            std_tensor = torch.tensor([std_tensor], dtype=torch.float32)
        # Ensure C, H, W format
        if std_tensor.dim() == 1:
            std_tensor = std_tensor.unsqueeze(-1)
        if std_tensor.dim() == 2:
            std_tensor = std_tensor.unsqueeze(0)
        features['std'] = std_tensor
        self.feature_dims['std'] = std_tensor.shape

        # Shape features
        shape_feats = self.shape_extractor.extract(images, masks)
        batch_size = images.shape[0]
        shape_tensor = torch.zeros((batch_size, len(self.shape_keys)), dtype=torch.float32)
        for i in range(batch_size):
            for j, key in enumerate(self.shape_keys):
                batch_key = f"{key}_{i}"
                if batch_key in shape_feats:
                    val = shape_feats[batch_key]
                elif key in shape_feats:
                    val = shape_feats[key]
                else:
                    val = 0.0
                shape_tensor[i, j] = float(val) if val is not None else 0.0
        # Each shape feature as (C=1, H=batch_size, W=1)
        for idx, key in enumerate(self.shape_keys):
            feature = shape_tensor[:, idx].unsqueeze(0).unsqueeze(-1)
            features[key] = feature
            self.feature_dims[key] = feature.shape

        # GLCM
        glcm_feats = self.glcm_extractor(images, masks)
        glcm_tensor = glcm_feats['glcm']
        if glcm_tensor.dim() == 2:
            glcm_tensor = glcm_tensor.unsqueeze(0)
        elif glcm_tensor.dim() == 1:
            glcm_tensor = glcm_tensor.unsqueeze(0).unsqueeze(-1)
        features['glcm'] = glcm_tensor
        self.feature_dims['glcm'] = glcm_tensor.shape

        # LBP
        lbp_feats = self.lbp_extractor(images, masks)
        lbp_tensor = lbp_feats['lbp']
        if lbp_tensor.dim() == 2:
            lbp_tensor = lbp_tensor.unsqueeze(0)
        elif lbp_tensor.dim() == 1:
            lbp_tensor = lbp_tensor.unsqueeze(0).unsqueeze(-1)
        features['lbp'] = lbp_tensor
        self.feature_dims['lbp'] = lbp_tensor.shape

        # Replace NaN values with 0 and print a warning if any NaNs are found
        nan_found = False
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
                features[key] = torch.nan_to_num(tensor, nan=0.0)
                nan_found = True
        return features
