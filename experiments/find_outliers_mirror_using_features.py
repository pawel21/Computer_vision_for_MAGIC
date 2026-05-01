import cv2
import h5py
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass

from MirrorFeatureExtractor.mirror_feature_extractor import (
    extract_glcm_features,
    extract_lbp_features,
    extract_edge_features,
    extract_features_for_mirror
)
from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor
from DetectionMetrics.detection_metrics import DetectionMetrics

# Feature keys in guaranteed order
GLCM_KEYS = [
    'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
    'glcm_energy', 'glcm_correlation',
]

LBP_KEYS = ['lbp_entropy', 'lbp_uniformity']
EDGE_KEYS = ['sobel_mean', 'laplacian_mean', 'laplacian_std', 'edge_density']

ALL_FEATURE_KEYS = GLCM_KEYS + LBP_KEYS + EDGE_KEYS

FEATURE_IDX = {key: i for i, key in enumerate(ALL_FEATURE_KEYS)}

N_MIRRORS = 249
N_FEATURES = len(ALL_FEATURE_KEYS)

MAD_SCALE = 1.4826  # konsystencja z odchyleniem standardowym przy rozkładzie normalnym
MAD_FLOOR = 1e-6    # zabezpieczenie przed dzieleniem przez 0

@dataclass
class Baseline:
    """Median and scale MAD per (mirros, feature)."""
    median: np.ndarray # shape (N_MIRRORS, N_FEATURES)
    mad: np.ndarray    # shape (N_MIRRORS, N_FEATURES)

    def z_scores(self, features: np.ndarray) -> np.ndarray:
        """Robust z-score (x - median) / (1.4826*MAD)"""
        return (features - self.median) / self.mad

def get_baseline(features: np.ndarray) -> Baseline:
    """
    Calculate robust baseline from N reference images
    """
    if features.ndim != 3:
        raise ValueError(
            f"Expected 3D array (n_images, n_mirrors, n_features), got shape {features.shape}"
        )
    # nanmedian — odporność na pojedyncze nieudane ekstrakcje
    median = np.nanmedian(features, axis=0)
    mad = np.nanmedian(np.abs(features - median[np.newaxis, :, :]), axis=0) * MAD_SCALE
    mad = np.maximum(mad, MAD_FLOOR)

    return Baseline(median=median, mad=mad)

with h5py.File("/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/baseline//baseline_webcam_features.h5", "r") as f:
    features = f["feature_matrix"][:]        # (n_images, 249, 11)



img_path = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/webcam_useful_image/webcam_useful_images/image_2024-05-05_1300.jpg"
mirror_extractor = SimpleMirrorExtractor("/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/calibration/points_WebCam.json")
img = cv2.imread(str(img_path))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
