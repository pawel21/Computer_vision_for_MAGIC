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

from vis_tools import mark_mirrors_on_img
from utils import load_config, get_features_array


def main():
    parser = argparse.ArgumentParser(description='Script for finding outlier mirrors in webcam images')
    parser.add_argument("--config", required=True, help="Configuration file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    BASELINE_FILE = cfg["paths"]["baseline"]
    img_path_list = cfg["paths"]["img_path_list"]
    CALIB_FILE = cfg["paths"]["calib"]
    print(img_path_list)
    features = get_features_array(BASELINE_FILE)
    baseline = build_vector_baseline(features)

    mirror_extractor = SimpleMirrorExtractor(CALIB_FILE)
    img_path = img_path_list[0]
    img = cv2.imread(str(img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #  Ekstrakcja z nowego obrazu
    new_features = extract_all_mirrors(img_gray, mirror_extractor)  # (249, 11)
if __name__ == "__main__":
    main()