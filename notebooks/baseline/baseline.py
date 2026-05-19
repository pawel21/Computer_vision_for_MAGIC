import numpy as np

from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.distance import mahalanobis

import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection, LineCollection

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

# Mapowanie nazwa -> indeks
GLCM_KEYS = ['glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
             'glcm_energy', 'glcm_correlation']
LBP_KEYS = ['lbp_entropy', 'lbp_uniformity']
EDGE_KEYS = ['sobel_mean', 'laplacian_mean', 'laplacian_std', 'edge_density']
ALL_FEATURE_KEYS = GLCM_KEYS + LBP_KEYS + EDGE_KEYS
N_FEATURES = len(ALL_FEATURE_KEYS)

FEATURE_IDX = {k: i for i, k in enumerate(ALL_FEATURE_KEYS)}

def build_hex_mirror_graph(row_sizes):
    """
    Buduje graf sąsiedztwa dla luster w układzie heksagonalnym.
    row_sizes: lista z liczbą luster w kolejnych rzędach,
               np. [14, 16, 18, 20, 22, 24, 26, 28, 26, 24, 22, 20, 18, 16, 14]
    """
    mirrors = {}  # (row, col) -> mirror_id
    positions = {}  # mirror_id -> (x, y) fizyczna pozycja
    mirror_id = 0

    max_width = max(row_sizes)

    for row_idx, n_mirrors in enumerate(row_sizes):
        # Offset - centrowanie każdego rzędu
        offset = (max_width - n_mirrors) / 2.0
        for col_idx in range(n_mirrors):
            mirrors[(row_idx, col_idx)] = mirror_id

            # Pozycja fizyczna (uwzględniając hex staggering)
            x = offset + col_idx
            y = row_idx * np.sqrt(3) / 2  # odstęp hex między rzędami
            positions[mirror_id] = (x, y)
            mirror_id += 1

    total_mirrors = mirror_id
    adjacency = defaultdict(list)

    for row_idx, n_mirrors in enumerate(row_sizes):
        for col_idx in range(n_mirrors):
            mid = mirrors[(row_idx, col_idx)]

            # Sąsiedzi w tym samym rzędzie
            if col_idx > 0:
                adjacency[mid].append(mirrors[(row_idx, col_idx - 1)])
            if col_idx < n_mirrors - 1:
                adjacency[mid].append(mirrors[(row_idx, col_idx + 1)])

            # Sąsiedzi w rzędzie powyżej i poniżej
            for d_row, neighbor_row_idx in [(-1, row_idx - 1), (1, row_idx + 1)]:
                if 0 <= neighbor_row_idx < len(row_sizes):
                    n_neighbor = row_sizes[neighbor_row_idx]
                    delta = (n_neighbor - n_mirrors)

                    # W siatce hex, sąsiedztwo zależy od tego,
                    # czy sąsiedni rząd jest szerszy czy węższy
                    if delta > 0:  # sąsiedni rząd szerszy
                        neighbor_cols = [col_idx, col_idx + 1]
                    elif delta < 0:  # sąsiedni rząd węższy
                        neighbor_cols = [col_idx - 1, col_idx]
                    else:  # ten sam rozmiar
                        neighbor_cols = [col_idx - 1, col_idx, col_idx + 1]

                    for nc in neighbor_cols:
                        if 0 <= nc < n_neighbor:
                            nid = mirrors[(neighbor_row_idx, nc)]
                            if nid not in adjacency[mid]:
                                adjacency[mid].append(nid)

    return adjacency, positions, total_mirrors



def make_square_patch(center, size):
    """Kwadratowe lustro wycentrowane na (center)."""
    return patches.Rectangle(
        (center[0] - size / 2, center[1] - size / 2), size, size
    )

def visualization_mirrors(feature_array, feature_name, ax):
    row_dict = {
        "row 1": 9,
        "row 2": 11,
        "row 3": 13,
        "row 4": 15,
        "row 5": 17,
        "row 6": 17,
        "row 7": 17,
        "row 8": 17,
        "row 9": 17,
        "row 10": 17,
        "row 11": 17,
        "row 12": 17,
        "row 13": 17,
        "row 14": 15,
        "row 15": 13,
        "row 16": 11,
        "row 17": 9,
    }

    list(row_dict.values())
    # Definicja rzędów (symetryczny sześciokąt)
    row_sizes = list(row_dict.values())
    adj, pos, n_total = build_hex_mirror_graph(row_sizes)
    sq_size = 0.85

    print(f"Łącznie luster: {n_total}")
    print(f"Lustro 50 -> sąsiedzi: {adj[1]}")

    # ──────────────────────────────────────────────
    # Wizualizacja 1: Graf sąsiedztwa + kwadraty
    # ──────────────────────────────────────────────

    edge_lines = []
    drawn_edges = set()
    for mid, neighbors in adj.items():
        for nid in neighbors:
            edge_key = tuple(sorted((mid, nid)))
            if edge_key not in drawn_edges:
                drawn_edges.add(edge_key)
                edge_lines.append([pos[mid], pos[nid]])

    lc = LineCollection(edge_lines, colors='#cccccc', linewidths=0.5,
                        zorder=1, alpha=0.6)
    ax.add_collection(lc)

    sq_patches = [make_square_patch(pos[mid], sq_size) for mid in range(n_total)]

    pc = PatchCollection(sq_patches, cmap='YlOrRd', edgecolors='#333333',
                         linewidths=0.8, zorder=2)

    pc.set_array(np.array(feature_array))

    pc.set_clim(min(feature_array), max(feature_array))
    ax.add_collection(pc)

    for mid in range(n_total):
        x, y = pos[mid]
        ax.text(x, y, str(mid), ha='center', va='center',
                fontsize=6.5, fontweight='bold', color='#222222', zorder=3)

    cbar = plt.colorbar(pc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(feature_name, fontsize=12)

    ax.set_xlim(-1, max(row_sizes) + 1)
    ax.set_ylim(16, -1)
    ax.set_aspect('equal')
    ax.set_title("MAGIC Mirrors")
    ax.set_xlabel('Column', fontsize=11)
    ax.set_ylabel('Row', fontsize=11)
    ax.grid(False)
    ax.set_facecolor('#f8f8f8')
    plt.tight_layout()

    plt.show()
