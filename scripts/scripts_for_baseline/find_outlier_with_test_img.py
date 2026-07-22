from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor

from MirrorFeatureExtractor.baseline import (
    build_vector_baseline,
    distance_mahalanobis,
    extract_all_mirrors
)
from DetectionMetrics.detection_metrics import DetectionMetrics, compute_metrics

from vis_tools import mark_mirrors_on_img, add_polygon_on_img
from utils import load_config, get_features_array

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("output")
WEBCAM_OUTPUT_DIR = Path("")
OUTLIER_THRESHOLD_STD = 3.5

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script for finding outlier mirrors in webcam images"
    )
    parser.add_argument("--config", required=True, type=Path, help="Configuration file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where annotated images are saved",
    )
    return parser.parse_args()

def find_outliers(scores: np.ndarray, n_std: float = OUTLIER_THRESHOLD_STD) -> np.ndarray:
    """Zwraca indeksy elementów, których wynik przekracza próg mean + n_std * std."""
    threshold = float(np.mean(scores) + n_std * np.std(scores))
    return np.where(scores > threshold)[0]

def process_image(
        img_path: Path,
        mirror_extractor: SimpleMirrorExtractor,
        baseline,
        ground_truth_mirror_id,
        output_dir: Path,
) -> None:
    """Wykrywa lusterka odstające na pojedynczym obrazie i zapisuje wizualizację."""
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning("Nie udało się wczytać obrazu: %s", img_path)
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    new_features = extract_all_mirrors(img_gray, mirror_extractor)
    scores_maha = distance_mahalanobis(new_features, baseline)
    outliers_maha = find_outliers(scores_maha)
    print(outliers_maha)
    logger.info(
        "%s: Predict number of outlier mirrors = %d (%s)",
        img_path.name,
        len(outliers_maha),
        outliers_maha.tolist(),
    )

    points = [mirror_extractor.get_point_coords(mirror_id) for mirror_id in outliers_maha]
    points_grouth_true = [mirror_extractor.get_point_coords(mirror_id) for mirror_id in ground_truth_mirror_id]
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].imshow(img_rgb)
    ax[1].imshow(img_rgb)
    add_polygon_on_img(ax[1], points, "red")
    add_polygon_on_img(ax[1], points_grouth_true, "green")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"checked_{img_path.name}"
    fig.savefig(out_path)
    plt.close(fig)

    logger.info("Zapisano wynik: %s", out_path)

    return outliers_maha

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    baseline_file = cfg["paths"]["baseline"]
    calib_file = cfg["paths"]["calib"]
    webcam_img_dir = Path(cfg["paths"]["img_path_dir"])
    json_path = cfg["paths"]["test_img"]
    with open(json_path, "r") as f:
        dict_test_img_list = json.load(f)

    features = get_features_array(baseline_file)
    baseline = build_vector_baseline(features)
    mirror_extractor = SimpleMirrorExtractor(calib_file)
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    for d in dict_test_img_list:
        img_path = webcam_img_dir / d["file"]
        ground_truth = d["marked_mirrors"]
        logger.info("Ścieżka: %s", img_path)
        if img_path.exists():
            logger.info("Przetwarzanie obrazu: %s", img_path)
            pred = process_image(img_path, mirror_extractor, baseline, ground_truth, args.output_dir)
            metrics = compute_metrics(pred, ground_truth)
            print(f"Result:        {metrics.summary}")
            tp_sum += metrics.tp
            fp_sum += metrics.fp
            fn_sum += metrics.fn

    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    logger.info(f"True positive: %d: {tp_sum}")
    logger.info(f"False positive: %d: {fp_sum}")
    logger.info(f"False negative: %d: {fn_sum}")
    logger.info("Summary precision: %.3f", precision)
    logger.info("Summary recall: %.3f", recall)

if __name__ == "__main__":
    main()