import cv2
import numpy as np
import h5py
import logging
import yaml

from scipy.ndimage import sobel, laplace, binary_erosion, binary_fill_holes, label, distance_transform_edt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

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


def get_features_array(path_to_h5_baseline_file):
    with h5py.File(path_to_h5_baseline_file, "r") as f:
        features = f["feature_matrix"][:]        # (n_images, 249, 11)
    return features