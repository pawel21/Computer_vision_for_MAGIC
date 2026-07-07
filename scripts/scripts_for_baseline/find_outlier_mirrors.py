import argparse

import cv2
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml

from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor

from MirrorExtractor.baseline import (
    VectorBaseline,
    build_vector_baseline,
    extract_all_mirrors,
    distance_mahalanobis,
    distance_euclidean,
    distance_cosine
)

from vis_tools import mark_mirrors_on_img, add_polygon_on_img

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_features_array(path_to_h5_baseline_file):
    with h5py.File(path_to_h5_baseline_file, "r") as f:
        features = f["feature_matrix"][:]        # (n_images, 249, 11)
    return features

def get_anomaly_index(baseline, new_features, feat_id):
    threshold_low = baseline.median[:, feat_id] - 2.5*baseline.std[:, feat_id ]
    threshold_high = baseline.median[:, feat_id] + 2.5*baseline.std[:, feat_id]
    value_array = new_features[:, feat_id]
    anomaly_index = np.where((value_array < threshold_low) | (value_array > threshold_high))
    print(anomaly_index[0])
    return anomaly_index[0]



# Mapowanie nazwa -> indeks
GLCM_KEYS = ['glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
             'glcm_energy', 'glcm_correlation']
LBP_KEYS = ['lbp_entropy', 'lbp_uniformity']
EDGE_KEYS = ['sobel_mean', 'laplacian_mean', 'laplacian_std', 'edge_density']
ALL_FEATURE_KEYS = GLCM_KEYS + LBP_KEYS + EDGE_KEYS


FEATURE_IDX = {k: i for i, k in enumerate(ALL_FEATURE_KEYS)}



def main():
    parser = argparse.ArgumentParser(description='Script for finding outlier mirrors in webcam images')
    parser.add_argument("--config", required=True, help="Configuration file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    BASELINE_FILE = cfg["paths"]["baseline"]
    img_path = cfg["paths"]["img_path"]
    CALIB_FILE = cfg["paths"]["calib"]
    features = get_features_array(BASELINE_FILE)
    baseline = build_vector_baseline(features)

    mirror_extractor = SimpleMirrorExtractor(CALIB_FILE)

    img = cv2.imread(str(img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #  Ekstrakcja z nowego obrazu
    new_features = extract_all_mirrors(img_gray, mirror_extractor)  # (249, 11)

    fig, ax = plt.subplots(3, 4, figsize=(12, 12))

    for i, key in enumerate(list(FEATURE_IDX.keys())):
        print(key)
        feat_id = FEATURE_IDX[key]
        outliers = get_anomaly_index(baseline, new_features, feat_id)
        p_list = []
        for m_id in outliers:
            p_list.append(mirror_extractor.get_point_coords(m_id))
        print(p_list)
        axes = ax.flatten()
        axes[i].imshow(img_rgb)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(key)
        add_polygon_on_img(axes[i], p_list, "red")

    # Liczymy odległość — score per lustro
    scores_maha = distance_mahalanobis(new_features, baseline)
    threshold_maha = np.mean(scores_maha) + 3 * np.std(scores_maha)
    print(threshold_maha)
    outliers_maha = np.where(scores_maha > threshold_maha)[0]
    print(outliers_maha)
    p_list_maha = []
    for m_id in outliers_maha:
        p_list_maha.append(mirror_extractor.get_point_coords(m_id))
    axes[11].imshow(img_rgb)
    add_polygon_on_img(axes[11], p_list_maha, "red")

    # axes[11].axis('off')
    plt.show()



if __name__ == "__main__":
    main()