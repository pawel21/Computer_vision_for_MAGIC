import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import logging
import os
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import List
import cv2
from pathlib import Path
from scipy.ndimage import sobel, laplace, binary_erosion, binary_fill_holes, label, distance_transform_edt

from sklearn.covariance import LedoitWolf


from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor
from MirrorFeatureExtractor.mirror_feature_extractor import (
    new_extract_glcm_features,
    new_extract_edge_features,
    new_extract_lbp_features
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def load_gray(img_path):
    img = cv2.imread(str(img_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def build_mask(gray, thr=0):
    m = gray > thr
    m = binary_fill_holes(m)  # ciemny piksel wewnątrz lustra wraca
    lab, n = label(m)
    if n > 1:  # zostaw największy spójny obszar
        sizes = np.bincount(lab.ravel());
        sizes[0] = 0
        m = lab == sizes.argmax()
    return m


def extract_photometric(gray, m):
    v = gray[m].astype(np.float64)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    p05, p25, p75, p95 = np.percentile(v, [5, 25, 75, 95])
    return {
        'median': med,
        'cov': v.std() / (v.mean() + 1e-8),
        'iqr_norm': (p75 - p25) / (med + 1e-8),
        'sat_frac': float((v >= 250).mean()),  # FLAGA, nie cecha
    }

def new_extract_edge_features(gray_img, mask):

    names = ['laplacian_std', 'edge_density']
    features = {n: np.nan for n in names}

    try:
        core = binary_erosion(mask, np.ones((3, 3), bool))
        if core.sum() < 10:
            return features

        # Replace background with the nearest in-mask value: no artificial step.
        idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        g = gray_img[tuple(idx)].astype(np.float64)

        lap = np.abs(laplace(g))
        features['laplacian_std'] = float(lap[core].std())

        edges = cv2.Canny(np.clip(g, 0, 255).astype(np.uint8), 50, 150)
        features['edge_density'] = float((edges[core] > 0).sum() / core.sum())

    except Exception as e:
        logging.warning(f"Edge feature extraction failed: {e}")

    return features

def proces_one_img(path, mirror_extractor):
    img_gray = load_gray(path)
    ph_array = np.zeros((249, 4))
    edge_array = np.zeros((249, 2))
    for m_id in range(249):
        m_gray = mirror_extractor.extract_mirror_gray(img_gray, mirror_id=m_id)
        mask = build_mask(m_gray)
        ph_dict = extract_photometric(m_gray, mask)
        ph_array[m_id, :] = list(ph_dict.values())
        edge_dict = new_extract_edge_features(m_gray, mask)
        edge_array[m_id, :] = list(edge_dict.values())
    feat_array = np.concatenate((ph_array, edge_array), axis=1)
    return feat_array

# ========== BASELINE (raz) ==========
def robust_z(x, axis, eps=1e-6):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
    return (x - med) / (1.4826 * mad + eps)

def mahalanobis_lw(X, mu, cols):
    Z = (X - mu).reshape(-1, len(cols))
    lw = LedoitWolf().fit(Z)
    return lw.mahalanobis(Z)

def d2_of(Xset, mu, lw):
    """Xset (N,249,5) -> d2 (N,249)"""
    Zr = robust_z(Xset, axis=1) - mu
    out = np.full(Zr.shape[:2], np.nan)
    g = np.isfinite(Zr).all(axis=2)
    out[g] = lw.mahalanobis(Zr[g])
    return out

if __name__ == '__main__':
    print("Start")

    mirror_extractor = SimpleMirrorExtractor("/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/points_WebCam.json")

    img_paths = Path("/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/MAGIC/webcam_useful_images")
    img_path_list = sorted(img_paths.glob("*.jpg"))
    print(len(img_path_list))

    N_IMG = 15
    X = np.zeros((N_IMG, 249, 6))
    for i, path in enumerate(img_path_list[:N_IMG]):
        img_gray = load_gray(path)
        feat_array = proces_one_img(path, mirror_extractor)
        X[i, :, :] = feat_array


    COLS = ['median', 'cov', 'iqr_norm', 'laplacian_std', 'edge_density']

    R_all = robust_z(X, axis=1)
    mu_all = np.nanmedian(R_all, axis=0)
    Zall = (R_all - mu_all).reshape(-1, len(COLS))
    Zall = Zall[np.isfinite(Zall).all(axis=1)]

    C = np.corrcoef(Zall.T)

    img_path = Path("/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/MAGIC/webcam_useful_images/image_2023-01-01_1200.jpg")
    x_new = proces_one_img(img_path, mirror_extractor)

    R = robust_z(X, axis=1)  # (N, 249, 5), axis=1 = oś luster
    mu = np.median(R, axis=0)  # (249, 5)
    sd = 1.4826 * np.median(np.abs(R - mu), axis=0)  # (249, 5)
    sd = np.maximum(sd, np.percentile(sd, 10))  # podłoga — inaczej dzielisz przez ~0
    z = robust_z(x_new, axis=0) - mu  # axis=0, bo (249, 5) — o wymiar mniej




    # podział: model uczy się na pierwszej połowie, próg mierzony na drugiej
    X_tr, X_cal = X[:10], X[10:]

    R_tr = robust_z(X_tr, axis=1)
    mu = np.nanmedian(R_tr, axis=0)  # (249, 5)

    # Use LedoifWolf
   # Zf = (R_tr - mu).reshape(-1, len(COLS))
   # lw = LedoitWolf().fit(Zf[np.isfinite(Zf).all(axis=1)])
   # d2 = np.full(249, np.nan)
   # g = np.isfinite(z).all(axis=1)
   # d2[g] = lw.mahalanobis(z[g])




