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

def extract_brightness_features(gray_img):
    features = {}
    mean = np.mean(gray_img)
    std = np.std(gray_img)
    features['brightness_mean'] = mean
    features['brightness_cov'] = std / (mean + 1e-8)
    return features