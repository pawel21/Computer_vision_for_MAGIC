import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern


def raw_pixels(image):
    return image.flatten() / 255.0

def hog_features(image):
    return hog(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)

def lbp_histogram(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
    return hist

def pca_features(images, n_components=10):
    flat = [img.flatten() / 255.0 for img in images]
    return PCA(n_components=n_components).fit_transform(flat)

def statistical_features(img):
    return np.array([
        np.mean(img),
        np.std(img),
        np.min(img),
        np.max(img),
        np.percentile(img, 25),
        np.percentile(img, 75),
    ])

