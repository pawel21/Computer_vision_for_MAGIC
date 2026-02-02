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