import argparse
import logging
import numpy as np
from pathlib import Path
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

from MirrorFeatureExtractor.baseline import (
    extract_photometric,
    new_extract_edge_features,
    proces_one_img,
    robust_z
)

from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor

from utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLS = ['median', 'cov', 'iqr_norm', 'laplacian_std', 'edge_density']

def main():
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


    img_path_list = sorted(img_path_dir.glob("*.jpg"))
    mirror_extractor = SimpleMirrorExtractor(CALIB_FILE)
    X = np.zeros((N_IMG, 249, 5))
    for i, path in enumerate(img_path_list[:N_IMG]):
        logger.info("Przetwarzanie obrazu: %s", path)
        feat_array = proces_one_img(path, mirror_extractor)
        X[i, :, :] = feat_array

    # trening
    R_tr = robust_z(X, axis=1)  # (N, 249, 5), po lustrach w klatce
    mu = np.nanmedian(R_tr, axis=0)  # (249, 5), baseline czasowy per lustro
    Z_tr = (R_tr - mu).reshape(-1, len(COLS))

    ok = ~np.isnan(Z_tr).any(axis=1)  # LedoitWolf nie obsługuje NaN
    lw = LedoitWolf().fit(Z_tr[ok])

    N_CAL = 30
    # --- kalibracja na osobnych klatkach ---
    X_cal = np.zeros((N_CAL, 249, len(COLS)))
    for i, path in enumerate(img_path_list[N_IMG:N_IMG + N_CAL]):
        X_cal[i] = proces_one_img(path, mirror_extractor)

    R_cal = robust_z(X_cal, axis=1)
    d2_cal = np.array([lw.mahalanobis(R_cal[i] - mu) for i in range(N_CAL)])  # (N_CAL, 249)

    d2_med = np.nanmedian(d2_cal, axis=0)  # (249,) typowe d2 dla każdego lustra
    d2_med = np.where(d2_med < 1e-6, np.nan, d2_med)

    ratio = d2_cal / d2_med  # znormalizowane, porównywalne między lustrami
    print("kwantyl 99.5 per lustro – rozrzut:",
          np.nanpercentile(ratio, 99.5, axis=0).round(1)[:20])
    thr = np.nanpercentile(ratio, 99.5)

    logger.info(f"próg empiryczny: {thr:.2f}   (chi2 99%: 15.1 na surowym d2)")

    np.savez(BASELINE_OUT_FILE, mu=mu, covariance=lw.covariance_, location=lw.location_,
             d2_med=d2_med, threshold=thr, cols=np.array(COLS),
             n_frames=N_IMG, n_calib=N_CAL, n_used=ok.sum())

if __name__ == "__main__":
    main()