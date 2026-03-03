import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import h5py
from pathlib import Path
from datetime import datetime

from PIL import Image as PILImage
import matplotlib.patches as patches

from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor
from MirrorFeatureExtractor.mirror_feature_extractor import MirrorFeatureExtractor

def mark_mirrors_on_img(img_path, points_list):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_rgb)

    for points in points_list:
        # === Polygon na podstawie 4 współrzędnych ===
        polygon = patches.Polygon(
                points,
                closed=True,
                linewidth=2,
                edgecolor='red',
                facecolor='red',
                alpha=0.3  # przezroczystość wypełnienia
            )
        ax.add_patch(polygon)

    plt.tight_layout()
    plt.show()

def get_feature_matrix_from_img(img_path):
    mirror_extractor = SimpleMirrorExtractor(str(MIRROR_POINTS_JSON))
    mirror_feature_extractor = MirrorFeatureExtractor()
    new_feature_matrix = np.zeros((249, 6))
    img_gray = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)
    for mirror_id in range(0, 249):
        mirror_crop = mirror_extractor.extract_mirror_gray(img_gray, mirror_id=mirror_id)
        feat = mirror_feature_extractor.extract_texture_features(mirror_crop)
        new_feature_matrix[mirror_id, :] = list(feat.values())
    return new_feature_matrix

def get_points_list(outliers):
    mirror_extractor = SimpleMirrorExtractor(str(MIRROR_POINTS_JSON))
    p_list = []
    for m_id in outliers:
        p_list.append(mirror_extractor.get_point_coords(m_id))
    return p_list

if __name__ == '__main__':
    BASE_DIR = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data"
    ROOT = Path(BASE_DIR) / "data/images_for_analysis"
    MIRROR_POINTS_JSON = Path(BASE_DIR) / "points_IRCam.json"
    BASELINE_FILE = Path(BASE_DIR) / "baseline_1.h5"

    IMG_PATH = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/data/2025/12/17/IRCamM1T20251217_081547M.jpg"
    IMG_PATH = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/data/2025/12/15/IRCamM1T20251215_084547M.jpg"
    IMG_PATH = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/data/2025/12/15/IRCamM1T20251215_081547M.jpg"

    # Wczytaj
    with h5py.File(BASELINE_FILE, 'r') as f:
        baseline_median = f['median'][:]
        baseline_mad = f['mad'][:]
        feature_names = f.attrs['feature_names']
        print(f"Baseline z {f.attrs['n_images']} zdjęć")
        print(feature_names)
    new_feature_matrix = get_feature_matrix_from_img(IMG_PATH)
    z_from_baseline = (new_feature_matrix - baseline_median) / baseline_mad

    # ['lbp_entropy' 'glcm_contrast' 'glcm_dissimilarity' 'glcm_homogeneity'
    #  'glcm_energy' 'glcm_correlation']
    # Odwróć (wysoki kontrast = źle)
    z_from_baseline[:, 0] *= -1  # lbp_entropy
    z_from_baseline[:, 1] *= -1  # glcm_contrast
    z_from_baseline[:, 2] *= -1  # glcm_dissimilarity

    composite = z_from_baseline.mean(axis=1)  # (249,)