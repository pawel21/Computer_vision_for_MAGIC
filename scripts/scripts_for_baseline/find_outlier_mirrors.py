import cv2
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
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

BASELINE_FILE_UBUNTU = "/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/baseline/baseline_webcam_features.h5"
BASELINE_FILE_WSL = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/baseline/baseline_webcam_features.h5"
BASELINE_FILE_WSL_2 = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/baseline/baseline_features.h5"

#%%
# Mapowanie nazwa -> indeks
GLCM_KEYS = ['glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
             'glcm_energy', 'glcm_correlation']
LBP_KEYS = ['lbp_entropy', 'lbp_uniformity']
EDGE_KEYS = ['sobel_mean', 'laplacian_mean', 'laplacian_std', 'edge_density']
ALL_FEATURE_KEYS = GLCM_KEYS + LBP_KEYS + EDGE_KEYS


FEATURE_IDX = {k: i for i, k in enumerate(ALL_FEATURE_KEYS)}

features = get_features_array(BASELINE_FILE_UBUNTU)
baseline = build_vector_baseline(features)

img_path = "/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/MAGIC/webcam_useful_images/image_2024-05-04_1700.jpg"
img_path = "/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/MAGIC/webcam_useful_images/image_2024-05-15_1800.jpg"
img_path = "/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/MAGIC/webcam_useful_images/image_2024-05-11_1700.jpg"
img_path = "/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/MAGIC/webcam_useful_images/image_2024-05-12_1000.jpg"
#img_path = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/data/webcam_img/image_2023-04-23_1400.jpg"

img = cv2.imread(str(img_path))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mirror_extractor = SimpleMirrorExtractor("/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/points_WebCam.json")

# 2. Ekstrakcja z nowego obrazu
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
axes[11].axis('off')
plt.show()