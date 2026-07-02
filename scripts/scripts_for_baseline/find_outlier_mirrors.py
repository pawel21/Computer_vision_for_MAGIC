import cv2
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor

from MirrorExtractor.baseline import VectorBaseline, build_vector_baseline

def get_features_array(path_to_h5_baseline_file):
    with h5py.File(path_to_h5_baseline_file, "r") as f:
        features = f["feature_matrix"][:]        # (n_images, 249, 11)
    return features

BASELINE_FILE_UBUNTU = "/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/baseline/baseline_webcam_features.h5"
BASELINE_FILE_WSL = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/baseline/baseline_webcam_features.h5"
BASELINE_FILE_WSL_2 = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/baseline/baseline_features.h5"

features = get_features_array(BASELINE_FILE_UBUNTU)
baseline = build_vector_baseline(features)