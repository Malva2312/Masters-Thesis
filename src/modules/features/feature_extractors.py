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
from src.modules.features._2D.frequency.wavelet import Wavelet
from src.modules.features._2D.intensity.fof import FirstOrderFeatures2D
from src.modules.features._2D.texture.haralick import Haralick

class FeatureExtractorManager:
    """
    Manages and applies all feature extractors to a batch of images and masks.
    Returns a dictionary with feature names as keys and torch tensors as values.
    """

    def __init__(self):
        self.shape_extractor = ShapeFeatures2D()
        self.glcm_extractor = GLCM()

        # Comment/uncomment extractors as needed
        #self.fft_extractor = FFTFeatureExtractor()
        self.gabor_extractor = Gabor()
        self.hog_extractor = HOGFeatureExtractor()
        #self.entropy_extractor = Entropy()
        #self.mean_extractor = Mean()
        #self.std_extractor = StandardDeviation()
        self.lbp_extractor = LBP()
        #self.wavelet_extractor = Wavelet()
        self.fof_extractor = FirstOrderFeatures2D()
        self.haralick_extractor = Haralick()

        # Feature keys
        self.shape_keys = self.shape_extractor.selected_features
        self.glcm_keys = ['glcm_' + prop for prop in self.glcm_extractor.props]

        # Feature dimensions (populated after extraction)
        self.feature_dims = {
            'fft_magnitude': None,
            'fft_phase': None,
            'wavelet': None,
            'gabor': None,
            'hog': None,
            'entropy': None,
            'mean': None,
            'std': None,
            'glcm': None,
            'lbp': None,
            'fof': None,
            'haralick': None
        }
        for key in self.shape_keys:
            self.feature_dims[key] = None
        for key in self.glcm_keys:
            self.feature_dims[key] = None

    def _to_tensor(self, arr):
        """Ensure arr is a torch.Tensor of dtype float32."""
        if isinstance(arr, torch.Tensor):
            return arr.float()
        return torch.tensor(arr, dtype=torch.float32)

    def _ensure_chw(self, tensor):
        """Ensure tensor is in (C, H, W) format."""
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def __call__(self, images: torch.Tensor, masks: torch.Tensor):
        features = {}

        # FFT
        if hasattr(self, 'fft_extractor'):
            fft_feats = self.fft_extractor(images, masks)
            for k in ['fft_magnitude', 'fft_phase']:
                features[k] = self._ensure_chw(self._to_tensor(fft_feats[k]))
                self.feature_dims[k] = features[k].shape

        # Gabor
        if hasattr(self, 'gabor_extractor'):
            gabor_feats = self.gabor_extractor(images, masks)
            features['gabor'] = self._ensure_chw(self._to_tensor(gabor_feats['gabor']))
            self.feature_dims['gabor'] = features['gabor'].shape

        # Wavelet
        if hasattr(self, 'wavelet_extractor'):
            wavelet_feats = self.wavelet_extractor(images, masks)
            features['wavelet'] = self._ensure_chw(self._to_tensor(wavelet_feats['wavelet']))
            self.feature_dims['wavelet'] = features['wavelet'].shape

        # HOG
        if hasattr(self, 'hog_extractor'):
            hog_feats = self.hog_extractor(images, masks)
            features['hog'] = self._ensure_chw(self._to_tensor(hog_feats['hog']))
            self.feature_dims['hog'] = features['hog'].shape

        # Entropy
        if hasattr(self, 'entropy_extractor'):
            entropy_feats = self.entropy_extractor(images, masks)
            features['entropy'] = self._ensure_chw(self._to_tensor(entropy_feats['entropy']))
            self.feature_dims['entropy'] = features['entropy'].shape

        # Mean
        if hasattr(self, 'mean_extractor'):
            mean_feats = self.mean_extractor(images, masks)
            features['mean'] = self._ensure_chw(self._to_tensor(mean_feats['mean']))
            self.feature_dims['mean'] = features['mean'].shape

        # Std
        if hasattr(self, 'std_extractor'):
            std_feats = self.std_extractor(images, masks)
            features['std'] = self._ensure_chw(self._to_tensor(std_feats['std']))
            self.feature_dims['std'] = features['std'].shape

        # First Order Features
        if hasattr(self, 'fof_extractor'):
            fof_feats = self.fof_extractor.extract(images, masks)
            for k, v in fof_feats.items():
                features[k] = self._ensure_chw(self._to_tensor(v))
                self.feature_dims[k] = features[k].shape
            features['fof'] = torch.stack(
                [features[k] for k in self.fof_extractor.selected_features], dim=-1
            ).view(images.shape[0], 1, -1)
            self.feature_dims['fof'] = features['fof'].shape

        # Shape features
        if hasattr(self, 'shape_extractor'):
            shape_feats = self.shape_extractor(images, masks)
            batch_size = images.shape[0]
            shape_tensor = torch.zeros((batch_size, len(self.shape_keys)), dtype=torch.float32)
            for i in range(batch_size):
                for j, key in enumerate(self.shape_keys):
                    batch_key = f"{key}_{i}"
                    val = shape_feats.get(batch_key, shape_feats.get(key, 0.0))
                    shape_tensor[i, j] = float(val) if val is not None else 0.0
            for idx, key in enumerate(self.shape_keys):
                features[key] = self._ensure_chw(shape_tensor[:, idx])
                self.feature_dims[key] = features[key].shape
            features['shape'] = shape_tensor.view(batch_size, 1, -1)
            self.feature_dims['shape'] = features['shape'].shape

        # GLCM and its properties
        if hasattr(self, 'glcm_extractor'):
            glcm_feats = self.glcm_extractor(images, masks)
            features['glcm'] = self._ensure_chw(self._to_tensor(glcm_feats['glcm']))
            self.feature_dims['glcm'] = features['glcm'].shape
            for prop in self.glcm_keys:
                if prop in glcm_feats:
                    features[prop] = self._ensure_chw(self._to_tensor(glcm_feats[prop]))
                    self.feature_dims[prop] = features[prop].shape

        # Haralick features
        if hasattr(self, 'haralick_extractor'):
            haralick_feats = self.haralick_extractor(images, masks)
            features['haralick'] = self._ensure_chw(self._to_tensor(haralick_feats['haralick']))
            self.feature_dims['haralick'] = features['haralick'].shape

        # LBP
        if hasattr(self, 'lbp_extractor'):
            lbp_feats = self.lbp_extractor(images, masks)
            features['lbp'] = self._ensure_chw(self._to_tensor(lbp_feats['lbp']))
            self.feature_dims['lbp'] = features['lbp'].shape

        # Replace NaN values with 0 and print a warning if any NaNs are found
        nan_found = False
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
                features[key] = torch.nan_to_num(tensor, nan=0.0)
                nan_found = True

        return features
