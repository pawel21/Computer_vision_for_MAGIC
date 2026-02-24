import glob
import os
import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from MirrorExtractor.mirror_extractor import MirrorExtractor

def extract_and_save_mirror(h5_file, mirror_id, img_path_list, extractor):
    """Ekstrahuje jedno lustro ze wszystkich obrazów i zapisuje do HDF5."""

    images = []
    source_files = []

    for path in tqdm(img_path_list, desc=f"Mirror {mirror_id:03d}", leave=False):
        try:
            img = np.array(Image.open(path).convert('RGB'))
            x_coords, y_coords = extractor.get_coords(mirror_id)
            cropped = extractor.extract_polygon_region_cv2(img, x_coords, y_coords)

            if cropped is not None:
                images.append(cropped)
                source_files.append(os.path.basename(path))
        except Exception as e:
            print(f"Błąd dla {path}, mirror {mirror_id}: {e}")
            continue

    if not images:
        print(f"Brak obrazów dla lustra {mirror_id}")
        return

    images_array = np.array(images, dtype=np.uint8)

    # Zapis do HDF5
    grp = h5_file.create_group(f'mirrors/{mirror_id:03d}')
    grp.create_dataset(
        'images',
        data=images_array,
        chunks=(1, *images_array.shape[1:]),
        compression='gzip',
        compression_opts=4
    )

    dt = h5py.special_dtype(vlen=str)
    grp.create_dataset('source_files', data=source_files, dtype=dt)

    grp.attrs['num_observations'] = len(images)


def extract_features(img):
    """Extract features optimized for MAGIC telescope mirror images."""
    features = {}
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Warm bright pixels (sun/flare signature)
        hue = hsv[:, :, 0].astype(float)
        sat = hsv[:, :, 1].astype(float)
        val = hsv[:, :, 2].astype(float)

        # ── Mask out central camera housing ──
        mask = np.ones((h, w), dtype=np.uint8) * 255
        cam_y, cam_x = int(h * 0.42), int(w * 0.65)
        cv2.circle(mask, (cam_x, cam_y), int(min(h, w) * 0.22), 0, -1)

        valid = gray[mask > 0]

        features['mean_brightness'] = np.mean(valid)
        features['std_brightness'] = np.std(valid)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['sharpness'] = np.std(laplacian)
        warn_pixel_ratio = get_warm_bright_ratio(img)
        features["warn_pixel_ratio"] = warn_pixel_ratio

        # General bright + saturated (any color)
        bright_saturated = (val > 200) & (sat > 80) & (mask > 0)
        features['bright_saturated_frac'] = np.sum(bright_saturated) / np.sum(mask > 0)
    except Exception as e:
        print(e)

    return features


def get_warm_bright_ratio(image):
    img = image[0:400, :]
    w, h = img.shape[0], img.shape[1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Warm bright pixels (sun/flare signature)
    hue = hsv[:, :, 0].astype(float)
    sat = hsv[:, :, 1].astype(float)
    val = hsv[:, :, 2].astype(float)

    # Lens flare is typically warm (H < 30 or H > 160 in OpenCV) AND bright AND saturated
    warm_bright = ((hue < 25) | (hue > 160)) & (val > 180) & (sat > 60)
    return (np.sum(warm_bright) / (w * h)) * 100