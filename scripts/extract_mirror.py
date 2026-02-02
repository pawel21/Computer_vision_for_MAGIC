import glob
import os

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


def extract_all_mirrors(h5_path, img_path_list, extractor, mirror_ids=None):
    """Ekstrahuje wybrane lustra i zapisuje do HDF5."""

    if mirror_ids is None:
        mirror_ids = range(200)

    with h5py.File(h5_path, 'w') as f:
        for mirror_id in tqdm(mirror_ids, desc="Ekstrakcja luster"):
            extract_and_save_mirror(f, mirror_id, img_path_list, extractor)


if __name__ == '__main__':
    BASE_DIR = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/"

    path_to_grid = os.path.join(BASE_DIR, "data/crossings_points_IRCamM1T20250702_161000M.pkl")
    images_dir = os.path.join(BASE_DIR, "data/data/2025/data_test")

    #path_to_grid = os.path.join(BASE_DIR, "data/crossings_points.pkl")
    #images_dir = os.path.join(BASE_DIR, "data/webcam_useful_image/webcam_useful_images/")

    h5_output = os.path.join(BASE_DIR, "data/data/2025/mirrors_dataset_test2.h5")

    extractor = MirrorExtractor(path_to_grid)
    img_list = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    print(f"Znaleziono {len(img_list)} obrazów")
    Mirrors_list = [
        4, 5, 6, 7, 8, 9, 10, 11, 12,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,

    ]
    extract_all_mirrors(h5_output, img_list, extractor, mirror_ids=Mirrors_list)

    # Sprawdzenie
    with h5py.File(h5_output, 'r') as f:
        for key in f['mirrors'].keys():
            print(f"Mirror {key}: {f[f'mirrors/{key}/images'].shape}")