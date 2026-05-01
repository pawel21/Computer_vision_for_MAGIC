import numpy as np

from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.distance import mahalanobis

from MirrorFeatureExtractor.mirror_feature_extractor import (
    extract_glcm_features,
    extract_lbp_features,
    extract_edge_features,
    extract_features_for_mirror
)

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
MAD_FLOOR = 1e-6    # zabezpieczenie przed dzieleniem przez 0

@dataclass
class Baseline:
    """Median and scale MAD per (mirros, feature)."""
    median: np.ndarray # shape (N_MIRRORS, N_FEATURES)
    mad: np.ndarray    # shape (N_MIRRORS, N_FEATURES)

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
    std: np.ndarray # for Euclidean distance
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

