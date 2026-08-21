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