import numpy as np
from skimage import feature

class LBPExtractor:
    def __init__(self, radius=1, n_points=8):
        self.radius = radius
        self.n_points = n_points

    def extract_lbp(self, image):
        if hasattr(image, "detach"):  # Check if image is a torch tensor
            image = image.detach().cpu().numpy()
        lbp = feature.local_binary_pattern(image, self.n_points, self.radius, method="uniform")
        return lbp

    def extract_features(self, images):
        features = []
        for image in images:
            lbp_features = self.extract_lbp(image)
            features.append(lbp_features)
        return np.array(features)