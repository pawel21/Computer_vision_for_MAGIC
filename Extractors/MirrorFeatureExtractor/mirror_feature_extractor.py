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


    @staticmethod
    def extract_texture_features(gray_img):
        """Extract texture features using LBP and GLCM"""
        features = {}

        # Local Binary Pattern (LBP) - chwyta lokalne textury
        lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
        features['lbp_mean'] = np.mean(lbp)
        features['lbp_std'] = np.std(lbp)

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
        features['sobel_std'] = np.std(sobel_magnitude)
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