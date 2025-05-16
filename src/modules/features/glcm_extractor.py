import numpy as np
from skimage.feature import greycomatrix, greycoprops

# Extractor for Gray Level Co-occurrence Matrix (GLCM) features
class GLCMExtractor: 
    def __init__(self, distances=[1], angles=[0], levels=256, symmetric=True, normed=True, properties=None):
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.symmetric = symmetric
        self.normed = normed
        if properties is None:
            self.properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        else:
            self.properties = properties

    def extract_glcm(self, image):
        if hasattr(image, "detach"):  # Check if image is a torch tensor
            image = image.detach().cpu().numpy()
        image = image.astype(np.uint8)
        glcm = greycomatrix(
            image,
            distances=self.distances,
            angles=self.angles,
            levels=self.levels,
            symmetric=self.symmetric,
            normed=self.normed
        )
        return glcm

    def extract_features(self, images):
        features = []
        for image in images:
            glcm = self.extract_glcm(image)
            feature_vector = []
            for prop in self.properties:
                vals = greycoprops(glcm, prop)
                feature_vector.extend(vals.flatten())
            features.append(feature_vector)
        return np.array(features)