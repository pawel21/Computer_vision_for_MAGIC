import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.distance import mahalanobis

import numpy as np
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection, LineCollection

from scipy.ndimage import sobel, laplace, binary_erosion, binary_fill_holes, label, distance_transform_edt

from MirrorFeatureExtractor.mirror_feature_extractor import (
    extract_glcm_features,
    extract_lbp_features,
    extract_edge_features,
    extract_brightness_features
)

from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor


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
    p25, p75 = np.percentile(v, [25, 75])
    return {
        'median': med,
        'cov': v.std() / (v.mean() + 1e-8),
        'iqr_norm': (p75 - p25) / (med + 1e-8)
    }


def new_extract_edge_features(gray_img, mask):
    names = ['laplacian_std', 'edge_density']
    features = {n: np.nan for n in names}

    try:
        core = binary_erosion(mask, np.ones((3, 3), bool))
        if core.sum() < 10:
            return features

        # Canny: gauss + histereza sięgają dalej niż 1 piksel -> większy margines.
        core_canny = binary_erosion(mask, np.ones((7, 7), bool))

        # Replace background with the nearest in-mask value: no artificial step.
        idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        g = gray_img[tuple(idx)].astype(np.float64)
        sx, sy = sobel(g, axis=1), sobel(g, axis=0)

        lap = np.abs(laplace(g))
        features['laplacian_std'] = float(lap[core].std())

        # Progi Canny'ego względne, nie 50/150 — inaczej jaśniejsze niebo
        # samo z siebie daje więcej krawędzi przy tej samej strukturze lustra.
        # L1, bo cv2.Canny domyślnie używa L1 (L2gradient=False).
        grad_l1 = np.abs(sx) + np.abs(sy)
        hi = float(np.percentile(grad_l1[core], 90))
        hi = max(hi, 10.0)  # płaski patch -> nie zjeżdżaj do zera
        lo = 0.4 * hi
        edges = cv2.Canny(g.astype(np.uint8), lo, hi)
        features['edge_density'] = (edges[core_canny] > 0).sum() / core_canny.sum()

    except Exception as e:
        logging.warning(f"Edge feature extraction failed: {e}")

    return features

def proces_one_img(path, mirror_extractor):
    img_gray = load_gray(path)
    ph_array = np.zeros((249, 3))
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

def robust_z(x, axis, min_scale=1e-8):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
    scale = 1.4826 * mad
    scale = np.where(scale < min_scale, np.inf, scale)  # brak zmienności -> z = 0
    return (x - med) / scale

# OLD CODE
FEATURE_REGISTRY = {
    "glcm": extract_glcm_features,
    "lbp": extract_lbp_features,
    "edge": extract_edge_features,
    "brightness": extract_brightness_features
}

def new_extract_features_for_mirror(img_gray, mirror_extractor, mirror_id,
                                  families=None):
    families = families or list(FEATURE_REGISTRY.keys())
    crop = mirror_extractor.extract_mirror_gray(img_gray, mirror_id=mirror_id)
    features = {}
    for name in families:
        features.update(FEATURE_REGISTRY[name](crop))
    return features

# Mapowanie nazwa -> indeks
GLCM_KEYS = ['glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
             'glcm_energy', 'glcm_correlation']
LBP_KEYS = ['lbp_entropy', 'lbp_uniformity']
EDGE_KEYS = ['sobel_mean', 'laplacian_mean', 'laplacian_std', 'edge_density']
ALL_FEATURE_KEYS = GLCM_KEYS + LBP_KEYS + EDGE_KEYS
N_FEATURES = len(ALL_FEATURE_KEYS)

FEATURE_IDX = {k: i for i, k in enumerate(ALL_FEATURE_KEYS)}

N_MIRRORS = 249
COV_REG = 1e-6

MAD_SCALE = 1.4826  # konsystencja z odchyleniem standardowym przy rozkładzie normalnym
MAD_FLOOR = 1e-6  # zabezpieczenie przed dzieleniem przez 0


def extract_features_for_mirror(
        img_gray: np.ndarray,
        mirror_extractor: SimpleMirrorExtractor,
        mirror_id: int,
) -> np.ndarray | None:
    """Extract all featues for a single mirror crop. Return None on failure."""
    try:
        crop = mirror_extractor.extract_mirror_gray(img_gray, mirror_id=mirror_id)
        glcm = extract_glcm_features(crop)
        lbp = extract_lbp_features(crop)
        edge = extract_edge_features(crop)

        vec = np.array(
            [glcm[k] for k in GLCM_KEYS]
            + [lbp[k] for k in LBP_KEYS]
            + [edge[k] for k in EDGE_KEYS],
            dtype=np.float32
        )
        return vec
    except Exception as e:
        print("Mirror %d extrraction failed: %s", mirror_id, e)
        return None



@dataclass
class Baseline:
    """Median and scale MAD per (mirros, feature)."""
    median: np.ndarray  # shape (N_MIRRORS, N_FEATURES)
    mad: np.ndarray  # shape (N_MIRRORS, N_FEATURES)

    def z_scores(self, features: np.ndarray) -> np.ndarray:
        """Robust z-score (x - median) / (1.4826*MAD)"""
        return (features - self.median) / self.mad


@dataclass
class VectorBaseline:
    """
    Baseline for vector features: for each mirrror hold median and covariance matrix
    calculated from N reference images.

    medain: (N_MIRRORS, n_selected_features)
    cov_inv: (N_MIRRORS, n_selected_features, n_selected_features)
    feature_keys: list of used features (for documentation/debug)
    feature_idx: index in fill matrix
    """
    median: np.ndarray
    cov_inv: np.ndarray
    std: np.ndarray  # for Euclidean distance
    feature_keys: list[str]
    feature_idx: np.ndarray


def select_features(
        features: np.ndarray,
        keys: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Select features from full matrix

    features: (N_IMAGES, N_MIRRORS, 11) lub (N_MIRRORS, 11)
    keys: features name list; None = all features

    Returns: (sliced_features, used_keys, idx_array)
    """
    if keys is None:
        keys = ALL_FEATURE_KEYS
    idx = np.array([FEATURE_IDX[k] for k in keys])
    return features[..., idx], list(keys), idx


def build_vector_baseline(
        features: np.ndarray,
        keys: Optional[list[str]] = None,
) -> VectorBaseline:
    """
    Build Vector baseline from N reference images.

    features: (N_IMAGES, N_MIRRORS, 11)
    keys: which feature keys to use (None = all)

    """
    sub, used_keys, idx = select_features(features, keys)
    n_images, n_mirrors, n_feat = sub.shape

    if n_images < n_feat + 2:
        raise ValueError(
            f"Za mało obrazów ({n_images}) do estymacji macierzy kowariancji "
            f"{n_feat}x{n_feat}. Potrzeba co najmniej {n_feat + 2}, "
            f"realistycznie 3-5x więcej."
        )

    median = np.nanmedian(sub, axis=0)  # (n_mirrors, n_feat)
    std = np.nanstd(sub, axis=0)
    std = np.maximum(std, 1e-6)

    # Macierz kowariancji per lustro
    cov_inv = np.zeros((n_mirrors, n_feat, n_feat))
    for i in range(n_mirrors):
        # Centrujemy względem mediany (robust), nie średniej
        x = sub[:, i, :] - median[i]
        # Maska na NaN (pomijamy obrazy z brakami)
        mask = ~np.any(np.isnan(x), axis=1)
        x = x[mask]
        if x.shape[0] < n_feat + 2:
            # Fallback: macierz diagonalna ze std
            cov = np.diag(std[i] ** 2)
        else:
            cov = np.cov(x, rowvar=False)
        # Regularyzacja Tichonowa — gwarantuje odwracalność
        cov += np.eye(n_feat) * COV_REG
        cov_inv[i] = np.linalg.inv(cov)

    return VectorBaseline(
        median=median,
        cov_inv=cov_inv,
        std=std,
        feature_keys=used_keys,
        feature_idx=idx,
    )


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



def extract_all_mirrors(img_gray, mirror_extractor) -> np.ndarray:
    """Ekstrakcja features dla wszystkich luster z jednego obrazu."""
    out = np.full((N_MIRRORS, N_FEATURES), np.nan)
    for i in range(N_MIRRORS):
        out[i, :] = extract_features_for_mirror(img_gray, mirror_extractor, i)
    return out


def distance_mahalanobis(
        new_features: np.ndarray,
        baseline: VectorBaseline,
) -> np.ndarray:
    """
    Mahalanobis dla każdego lustra.

    new_features: (N_MIRRORS, 11) — pełen wektor z nowego obrazu
    Returns: (N_MIRRORS,) — dystans per lustro
    """
    sub = new_features[:, baseline.feature_idx]  # (n_mirrors, n_feat)
    diff = sub - baseline.median  # (n_mirrors, n_feat)

    # Wektoryzacja: diff @ cov_inv @ diff.T per lustro
    # Używamy einsum: 'ij, ijk, ik -> i'
    d2 = np.einsum('ij,ijk,ik->i', diff, baseline.cov_inv, diff)
    return np.sqrt(np.maximum(d2, 0))  # zabezpieczenie przed numerycznym <0


def distance_euclidean(
        new_features: np.ndarray,
        baseline: VectorBaseline,
        standardize: bool = True,
) -> np.ndarray:
    """
    Euclidean (opcjonalnie po standaryzacji per cecha).
    """
    sub = new_features[:, baseline.feature_idx]
    diff = sub - baseline.median
    if standardize:
        diff = diff / baseline.std
    return np.linalg.norm(diff, axis=1)


def distance_cosine(
        new_features: np.ndarray,
        baseline: VectorBaseline,
) -> np.ndarray:
    """
    Cosine distance między wektorem cech a medianą baseline'u.
    """
    sub = new_features[:, baseline.feature_idx]
    median = baseline.median

    dot = np.sum(sub * median, axis=1)
    norm_sub = np.linalg.norm(sub, axis=1)
    norm_med = np.linalg.norm(median, axis=1)
    cos_sim = dot / (norm_sub * norm_med + 1e-12)
    return 1.0 - cos_sim


# Mapowanie nazwa -> indeks
GLCM_KEYS = ['glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
             'glcm_energy', 'glcm_correlation']
LBP_KEYS = ['lbp_entropy', 'lbp_uniformity']
EDGE_KEYS = ['sobel_mean', 'laplacian_mean', 'laplacian_std', 'edge_density']
ALL_FEATURE_KEYS = GLCM_KEYS + LBP_KEYS + EDGE_KEYS
N_FEATURES = len(ALL_FEATURE_KEYS)

FEATURE_IDX = {k: i for i, k in enumerate(ALL_FEATURE_KEYS)}









