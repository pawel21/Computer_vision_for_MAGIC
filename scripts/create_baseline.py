import argparse
import datetime
import logging
import os
import sys
import glob
import numpy as np
import h5py
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from MirrorFeatureExtractor.mirror_feature_extractor import (
    extract_glcm_features,
    extract_lbp_features,
    extract_edge_features
)

from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)

# Feature keys in guaranteed order
GLCM_KEYS = [
    'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
    'glcm_energy', 'glcm_correlation',
]

LBP_KEYS = ['lbp_entropy', 'lbp_uniformity']
EDGE_KEYS = ['sobel_mean', 'laplacian_mean', 'laplacian_std', 'edge_density']

ALL_FEATURE_KEYS = GLCM_KEYS + LBP_KEYS + EDGE_KEYS

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def collect_image_path(glob_pattern: str, max_images: int | None = None) -> list[str]:
    "Return sorted image paths, optionally capped at max_images"
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f'No images found matching: {glob_pattern}')
    if max_images is not None:
        paths = paths[:max_images]
    logger.info(f"found %d images (using %d)", len(sorted(glob.glob(glob_pattern))), len(paths))
    return paths

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
        logger.warning("Mirror %d extrraction failed: %s", mirror_id, e)
        return None

def build_feature_matrix(
        image_paths: list[str],
        mirror_extractor: SimpleMirrorExtractor,
        n_mirrors: int = 249,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (n_images, n_mirrors, n_features) matrix

    Returns
    ------
    feature_matrix : np.ndarray, shape (n_images, n_mirrors, n_features)
    valid_mask     : np.ndarray, shape (n_images, n_mirrors), bool
        True where extraction succeeded
    """
    n_features = len(ALL_FEATURE_KEYS)
    n_images = len(image_paths)

    feature_matrix = np.full((n_images, n_mirrors, n_features), np.nan, dtype=np.float32)
    valid_mask = np.zeros((n_images, n_mirrors), dtype=np.bool)

    for i, path in enumerate(tqdm(image_paths, desc="Processing images")):
        img = cv2.imread(str(path))
        if img is None:
            logger.warning("Could not read image: %s", path)
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for mirror_id in range(n_mirrors):
            vec = extract_features_for_mirror(img_gray, mirror_extractor, mirror_id)
            if vec is not None:
                feature_matrix[i, mirror_id, :] = vec
                valid_mask[i, mirror_id] = True

    n_valid = valid_mask.sum()
    n_total = n_images * n_mirrors
    logger.info(
        "Extraction complete: %d / %d valid (%.1f%%)",
        n_valid, n_total, 100 * n_valid / n_total,
    )
    return feature_matrix, valid_mask

def save_baseline_hdf5(
        output_path: str,
        feature_matrix: np.ndarray,
        valid_mask: np.ndarray,
        image_paths: list[str],
        config: dict
) -> None:
    """Save feature matrix + full metadata to HDF5"""

    with h5py.File(output_path, 'w') as f:
        f.create_dataset(
            'feature_matrix',
            data=feature_matrix,
            compression='gzip',
            compression_opts=4,
        )
        f.create_dataset("valid_mask", data=valid_mask, compression='gzip')

        # ── feature names (fixed-length ASCII for portability) ──────
        dt = h5py.string_dtype("ascii", 64)
        f.create_dataset('features_name', data=ALL_FEATURE_KEYS, dtype=dt)

        # ── image paths used ────────────────────────────────────────
        dt_path = h5py.string_dtype("utf-8", 512)
        f.create_dataset(
            "image_paths",
            data=[str(p) for p in image_paths],
            dtype=dt_path,
        )
        # also store just filenames for quick reference
        f.create_dataset(
            "image_filenames",
            data=[Path(p).name for p in image_paths],
            dtype=dt_path,
        )

        # ── global metadata as HDF5 attributes ─────────────────────
        f.attrs["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        f.attrs["n_images"] = feature_matrix.shape[0]
        f.attrs["n_mirrors"] = feature_matrix.shape[1]
        f.attrs["n_features"] = feature_matrix.shape[2]
        f.attrs["config_yaml"] = yaml.dump(config)
        f.attrs["mirror_points_path"] = config["paths"]["mirror_points"]

    logger.info("Saved baseline to %s (%.1f MB)", output_path, Path(output_path).stat().st_size / 1e6)


def main():
    parser = argparse.ArgumentParser(description="Create mirror feature baseline HDF5.")
    parser.add_argument('--config', required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="List images and exit.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    image_paths = collect_image_path(
        cfg["paths"]["image_glob"],
        max_images=cfg["extraction"].get("max_images"),
    )

    if args.dry_run:
        print(f"Would process {len(image_paths)} images:")
        for p in image_paths[:10]:
            print(f"  {p}")
        if len(image_paths) > 10:
            print(f"  ... and {len(image_paths) - 10} more")
        sys.exit(0)

    mirror_extractor = SimpleMirrorExtractor(cfg["paths"]["mirror_points"])

    feature_matrix, valid_mask = build_feature_matrix(
        image_paths,
        mirror_extractor,
        n_mirrors=cfg["extraction"].get("n_mirrors", 249),
    )

    save_baseline_hdf5(
        output_path=cfg["paths"]["output_file"],
        feature_matrix=feature_matrix,
        valid_mask=valid_mask,
        image_paths=image_paths,
        config=cfg,
    )


if __name__ == '__main__':
    main()

