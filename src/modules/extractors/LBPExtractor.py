import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for LBP
class LBPExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, radius=1, n_points=8, method='uniform'):
        self.radius = radius
        self.n_points = n_points
        self.method = method

    def fit(self, _X, _y=None):
        return self

    def transform(self, X):
        lbp_features = []
        for image in X:
            lbp = local_binary_pattern(image.numpy(), self.n_points, self.radius, self.method)
            # Histogram of LBP
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, self.n_points + 3),
                                     range=(0, self.n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            lbp_features.append(hist)
        return np.array(lbp_features)
