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
            #'PerimeterSurfaceRatio',
            'Sphericity',
            #'SphericalDisproportion',
            #'Maximum2DDiameter'
        ]

        self.feature_dims = {
            'fft_magnitude': None,
            'fft_phase': None,
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

        # HOG
        hog_feats = self.hog_extractor(images, masks)
        hog_tensor = torch.tensor(hog_feats['hog'], dtype=torch.float32)
        features['hog'] = hog_tensor
        self.feature_dims['hog'] = hog_tensor.shape

        # Entropy
        entropy_feats = self.entropy_extractor(images, masks)
        entropy_tensor = entropy_feats['entropy']
        if not isinstance(entropy_tensor, torch.Tensor):
            entropy_tensor = torch.tensor([entropy_tensor], dtype=torch.float32)
        features['entropy'] = entropy_tensor if entropy_tensor.dim() > 0 else entropy_tensor.unsqueeze(0)
        self.feature_dims['entropy'] = features['entropy'].shape

        # Mean
        mean_feats = self.mean_extractor(images, masks)
        mean_tensor = mean_feats['mean']
        if isinstance(mean_tensor, list):
            mean_tensor = torch.tensor(mean_tensor, dtype=torch.float32)
        else:
            mean_tensor = torch.tensor([mean_tensor], dtype=torch.float32)
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
        for idx, key in enumerate(self.shape_keys):
            features[key] = shape_tensor[:, idx]
            self.feature_dims[key] = features[key].shape

        # GLCM
        glcm_feats = self.glcm_extractor(images, masks)
        features['glcm'] = glcm_feats['glcm']
        self.feature_dims['glcm'] = features['glcm'].shape

        # LBP
        lbp_feats = self.lbp_extractor(images, masks)
        features['lbp'] = lbp_feats['lbp']
        self.feature_dims['lbp'] = features['lbp'].shape
        # Replace NaN values with 0 and print a warning if any NaNs are found
        nan_found = False
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
                features[key] = torch.nan_to_num(tensor, nan=0.0)
                nan_found = True
                #print(f"Warning: NaN values found in feature '{key}'. Replaced with 0.")
        return features
