import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import logging
import os
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
from pathlib import Path
from scipy.ndimage import sobel, laplace, binary_erosion, binary_fill_holes, label, distance_transform_edt

from sklearn.covariance import LedoitWolf

from utils import load_config, proces_one_img, robust_z

from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor

COLS = ['median', 'cov', 'iqr_norm', 'laplacian_std', 'edge_density']

MY_THRESHOLD = 15

def mark_mirrors_on_img(img_path, points_list, color):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(img_rgb)

    for points in points_list:
        # === Polygon na podstawie 4 współrzędnych ===
        polygon = patches.Polygon(
                points,
                closed=True,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.3  # przezroczystość wypełnienia
            )
        ax.add_patch(polygon)

    plt.tight_layout()
    plt.show()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for create baseline file')
    parser.add_argument("--config", required=True, help="Configuration file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    BASELINE_OUT_FILE = cfg["output"]["baseline_file"]
    img_path_dir = Path(cfg["paths"]["img_path_dir"])
    CALIB_FILE = cfg["paths"]["calib"]
    N_IMG = cfg["parameters"]["number_of_image_to_process"]
    logger.info(f"Baseline: {BASELINE_OUT_FILE}")
    logger.info(f"Calib: {CALIB_FILE}")
    logger.info(f"Img path dir: {img_path_dir}")
    logger.info(f"Number of images to process: {N_IMG}")


    img_path = Path(
         "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/webcam/2024/05/good/20240507_1140_la.jpg")
    mirror_extractor = SimpleMirrorExtractor(CALIB_FILE)

    X_new = proces_one_img(img_path, mirror_extractor)
    R_new = robust_z(X_new[None, ...], axis=1)[0]

    b = np.load(BASELINE_OUT_FILE, allow_pickle=False)

    assert list(b["cols"]) == COLS, "inny zestaw cech niż przy budowaniu baseline'u"

    lw = LedoitWolf()
    lw.covariance_ = b["covariance"]
    lw.location_ = b["location"]
    lw.precision_ = np.linalg.pinv(b["covariance"])

    X_new = proces_one_img(img_path, mirror_extractor)
    R_new = robust_z(X_new[None, ...], axis=1)[0]
    Z_new = R_new - b["mu"]
    d2 = lw.mahalanobis(Z_new)

    d2 = lw.mahalanobis(Z_new)
    score = d2 / b["d2_med"]
    outliers_mirrors = np.where(score > b["threshold"])[0]

    points = [mirror_extractor.get_point_coords(m_id) for m_id in outliers_mirrors]

    mark_mirrors_on_img(img_path, points, "red")




