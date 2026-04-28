import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.path import Path
from PIL import Image
import cv2
import pickle 
import glob
import os

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.ndimage import sobel, laplace
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.decomposition import PCA

from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor


class MirrorFeatureExtractor:
    """Extracts features from mirror images"""

    @staticmethod
    def extract_brightness_features(mirror_img):
        """Extract brightness features"""
        features = {}

        # Brightness features
        features['brightness_mean'] = np.mean(mirror_img)
        features['brightness_std'] = np.std(mirror_img)
        features['brightness_min'] = np.min(mirror_img)
        features['brightness_max'] = np.max(mirror_img)
        features['brightness_range'] = np.max(mirror_img) - np.min(mirror_img)
        return features

    def extract_lpb_features(self, gray_img):
        """Extract LPB features"""
        features = {}

        # Local Binary Pattern (LBP) - chwyta lokalne textury
        lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
        features['lbp_mean'] = np.mean(lbp)
        features['lbp_std'] = np.std(lbp)

        # Entropy tekstury
        hist, _ = np.histogram(lbp, bins=59, range=(0, 59), density=True)
        features['lbp_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))

    @staticmethod
    def extract_texture_features(gray_img):
        """Extract texture features using LBP and GLCM"""
        features = {}

        # Local Binary Pattern (LBP) - chwyta lokalne textury
        lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')

        # Entropy tekstury
        hist, _ = np.histogram(lbp, bins=59, range=(0, 59), density=True)
        features['lbp_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))

        # Gray-level co-occurrence matrix (GLCM) - dla bardziej zaawansowanej analizy
        try:
            glcm = graycomatrix(gray_img.astype(np.uint8), distances=[1], angles=[0], levels=256)
            features['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
            features['glcm_dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
            features['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
            features['glcm_energy'] = graycoprops(glcm, 'energy')[0, 0]
            features['glcm_correlation'] = graycoprops(glcm, 'correlation')[0, 0]
            features['glcm_asm'] = graycoprops(glcm, 'asm')[0, 0]
        except:
            pass

        return features

    @staticmethod
    def extract_edge_and_gradient_features(gray_img):
        """Extract edge and gradient features"""
        features = {}

        # Sobel edges
        sobelx = sobel(gray_img, axis=1)
        sobely = sobel(gray_img, axis=0)
        sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        features['sobel_mean'] = np.mean(sobel_magnitude)
        features['sobel_max'] = np.max(sobel_magnitude)

        # Laplacian - dla detekcji zmian drugiego rzędu
        laplacian = laplace(gray_img)
        features['laplacian_mean'] = np.mean(np.abs(laplacian))
        features['laplacian_std'] = np.std(np.abs(laplacian))

        # Canny edge detection
        edges = cv2.Canny(gray_img.astype(np.uint8), 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size

        return features

    @staticmethod
    def extract_histogram_features(img):
        """Extract histogram-based features"""
        features = {}

        # Histogram dla każdego kanału
        for i, channel in enumerate(['R', 'G', 'B']):
            hist, _ = np.histogram(img[:, :, i], bins=256, range=(0, 256), density=True)
            features[f'{channel}_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
            features[f'{channel}_skewness'] = skew(hist)
            features[f'{channel}_kurtosis'] = kurtosis(hist)

        return features

    @staticmethod
    def extract_all_features(mirror_img):
        """Extract all features from a single mirror image"""
        features = {}

        # Brightness features
        features['brightness_mean'] = np.mean(mirror_img)
        features['brightness_std'] = np.std(mirror_img)
        features['brightness_min'] = np.min(mirror_img)
        features['brightness_max'] = np.max(mirror_img)
        features['brightness_range'] = np.max(mirror_img) - np.min(mirror_img)

        # Channel-specific features
        for i, channel in enumerate(['R', 'G', 'B']):
            features[f'{channel}_mean'] = np.mean(mirror_img[:, :, i])
            features[f'{channel}_std'] = np.std(mirror_img[:, :, i])

        # Statistical features
        features['skewness'] = skew(mirror_img.flatten())
        features['kurtosis'] = kurtosis(mirror_img.flatten())

        # Texture features
        gray = cv2.cvtColor(mirror_img, cv2.COLOR_RGB2GRAY)
        features.update(MirrorFeatureExtractor.extract_texture_features(gray))

        # Edge/Gradient features
        features.update(MirrorFeatureExtractor.extract_edge_features(gray))

        # Histogram features
        features.update(MirrorFeatureExtractor.extract_histogram_features(mirror_img))

        return features

GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']


def extract_glcm_features(gray_img):
    """Extract texture features using Gray-Level Co-occurrence Matrix (GLCM).

    Computes GLCM at distance=1 across four angles (0°, 45°, 90°, 135°)
    and averages the results to produce rotation-invariant features.

    Features extracted:
        - glcm_contrast: intensity difference between neighboring pixels.
          Higher = rougher texture, potential surface degradation.
        - glcm_dissimilarity: similar to contrast but linear weighting.
          Higher = more local variation in reflection.
        - glcm_homogeneity: closeness of GLCM values to diagonal.
          Higher = more uniform reflection (smooth mirror).
        - glcm_energy: sum of squared GLCM entries.
          Higher = more repetitive/ordered texture pattern.
        - glcm_correlation: linear dependency of gray levels on neighbors.
          Higher = more predictable spatial structure.

    Args:
        gray_img: 2D numpy array, grayscale mirror patch (typically ~80x90 px).
                  Will be cast to uint8 internally for GLCM computation.

    Returns:
        dict: Feature name -> float value. On failure, all values are np.nan.
    """
    features = {}

    try:
        angles_list = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(
            gray_img.astype(np.uint8),
            distances=[1],
            angles=angles_list,
            levels=256)
        for prop in GLCM_PROPERTIES:
            features[f"glcm_{prop}"] = graycoprops(glcm, prop).mean()

    except Exception as e:
        logging.warning(f"GLCM extraction failed with exception: {e}")
        for prop in GLCM_PROPERTIES:
            features[f"glcm_{prop}"] = np.nan

    return features


def extract_lbp_features(gray_img, points=8, radius=1):
    """Extract texture features using Local Binary Pattern (LBP).

    Computes uniform LBP and derives entropy and uniformity metrics
    from the pattern histogram. Uniform LBP captures local micro-textures
    (edges, corners, flat regions) — for mirror patches, higher entropy
    indicates more complex/irregular reflection texture (potential degradation),
    while higher uniformity suggests smooth, consistent reflection.

    Features extracted:
        - lbp_entropy: Shannon entropy of the LBP histogram.
          Higher = more diverse local patterns, less uniform surface.
        - lbp_uniformity: max bin proportion in the histogram.
          Higher = one dominant pattern (typically flat regions on good mirrors).

    Args:
        gray_img: 2D numpy array, grayscale mirror patch (typically ~80x90 px).
        points: number of neighbors in LBP circle (default 8).
        radius: radius of LBP circle in pixels (default 1).

    Returns:
        dict: Feature name -> float value. On failure, all values are np.nan.
    """
    n_bins = points * (points - 1) + 3  # uniform LBP: P*(P-1)+3 unique patterns
    features = {}

    try:
        lbp = local_binary_pattern(gray_img, P=points, R=radius, method='uniform')

        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

        # Shannon entropy — miara złożoności tekstury
        features['lbp_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))

        # Uniformity — jak bardzo dominuje jeden wzorzec
        features['lbp_uniformity'] = hist.max()

    except Exception as e:
        logging.warning(f"LBP extraction failed: {e}")
        features['lbp_entropy'] = np.nan
        features['lbp_uniformity'] = np.nan

    return features


def extract_edge_features(gray_img):
    """Extract edge and gradient features using Sobel, Laplacian, and Canny.

    Measures sharpness and structure of reflected image on mirror patches.
    Sharp, clear reflections indicate good mirror condition — degraded or
    dirty mirrors produce blurred, low-contrast reflections with weaker
    edge responses.

    Features extracted:
        - sobel_mean: mean gradient magnitude (Sobel).
          Higher = sharper reflection with more defined edges.
        - laplacian_mean: mean absolute second derivative.
          Higher = more rapid intensity changes, sharper details.
        - laplacian_std: std of absolute second derivative.
          Higher = more variation in sharpness across the patch.
        - edge_density: fraction of pixels detected as edges (Canny).
          Higher = more structural detail visible in reflection.

    Args:
        gray_img: 2D numpy array, grayscale mirror patch (typically ~80x90 px).
                  Will be cast to uint8 internally for Canny detection.

    Returns:
        dict: Feature name -> float value. On failure, all values are np.nan.
    """
    feature_names = ['sobel_mean', 'laplacian_mean', 'laplacian_std', 'edge_density']
    features = {}

    try:
        # Sobel — gradient pierwszego rzędu
        sobelx = sobel(gray_img, axis=1)
        sobely = sobel(gray_img, axis=0)
        sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        features['sobel_mean'] = np.mean(sobel_magnitude)

        # Laplacian — gradient drugiego rzędu, wrażliwy na rozmycie
        lap = np.abs(laplace(gray_img))
        features['laplacian_mean'] = np.mean(lap)
        features['laplacian_std'] = np.std(lap)

        # Canny — binarna mapa krawędzi, gęstość = ile struktury widać
        edges = cv2.Canny(gray_img.astype(np.uint8), 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size

    except Exception as e:
        logging.warning(f"Edge feature extraction failed: {e}")
        for name in feature_names:
            features[name] = np.nan

    return features

def extract_features_for_mirror(
        img_gray: np.ndarray,
        mirror_extractor: SimpleMirrorExtractor,
        mirror_id: int,
) -> np.ndarray | None:
    """Extract all featues for a single mirror crop. Return None on failure."""
    try:
        crop = mirror_extractor.extract_mirror_gray(img_gray, mirror_id=mirror_id)
        glcm = extract_glcm_features(crop)
        lbp = extract_lbp_features(crop)
        edge = extract_edge_features(crop)

        vec = np.array(
            [glcm[k] for k in GLCM_KEYS]
            + [lbp[k] for k in LBP_KEYS]
            + [edge[k] for k in EDGE_KEYS],
            dtype=np.float32
        )
        return vec
    except Exception as e:
        print("Mirror %d extrraction failed: %s", mirror_id, e)
        return None