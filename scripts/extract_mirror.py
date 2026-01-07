import glob
import os
import re
from datetime import datetime

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from MirrorExtractor.mirror_extractor import MirrorExtractor


def parse_timestamp_from_filename(filename):
    """Wyciąga timestamp z nazwy pliku: image_2023-01-01_1200.jpg -> datetime"""
    basename = os.path.basename(filename)
    match = re.search(r'image_(\d{4}-\d{2}-\d{2})_(\d{4})\.jpg', basename)
    if match:
        date_str, time_str = match.groups()
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H%M")
        return dt
    return None


def create_hdf5_dataset(h5_path, num_mirrors=200):
    """Tworzy strukturę HDF5 dla wszystkich luster."""
    with h5py.File(h5_path, 'w') as f:
        # Metadane globalne
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['num_mirrors'] = num_mirrors

        # Grupa dla każdego lustra
        for mirror_id in range(num_mirrors):
            grp = f.create_group(f'mirrors/{mirror_id:03d}')
            # Datasety będą tworzone dynamicznie przy pierwszym zapisie
            # (bo nie znamy rozmiaru wyciętego lustra z góry)
            grp.attrs['mirror_id'] = mirror_id

    print(f"Utworzono strukturę HDF5: {h5_path}")


def extract_and_save_mirror(h5_file, mirror_id, img_path_list, extractor):
    """Ekstrahuje jedno lustro ze wszystkich obrazów i zapisuje do HDF5."""

    grp_path = f'mirrors/{mirror_id:03d}'

    # Sortuj obrazy po timestamp
    img_path_list = sorted(img_path_list, key=lambda x: parse_timestamp_from_filename(x) or datetime.min)

    images = []
    timestamps = []
    source_files = []

    for path in tqdm(img_path_list, desc=f"Mirror {mirror_id:03d}", leave=False):
        try:
            img = np.array(Image.open(path).convert('RGB'))
            x_coords, y_coords = extractor.get_coords(mirror_id)
            cropped = extractor.extract_polygon_region_cv2(img, x_coords, y_coords)

            ts = parse_timestamp_from_filename(path)
            if ts and cropped is not None:
                images.append(cropped)
                timestamps.append(ts.timestamp())  # Unix timestamp
                source_files.append(os.path.basename(path))
        except Exception as e:
            print(f"Błąd dla {path}, mirror {mirror_id}: {e}")
            continue

    if not images:
        print(f"Brak obrazów dla lustra {mirror_id}")
        return

    # Konwersja do numpy arrays
    # Uwaga: jeśli lustra mają różne rozmiary, trzeba je znormalizować
    # lub zapisać jako variable-length dataset
    images_array = np.array(images, dtype=np.uint8)
    timestamps_array = np.array(timestamps, dtype=np.float64)

    # Zapis do HDF5
    grp = h5_file[grp_path]

    # Obrazy z kompresją
    grp.create_dataset(
        'images',
        data=images_array,
        chunks=(1, *images_array.shape[1:]),  # chunk = 1 obraz
        compression='gzip',
        compression_opts=4
    )

    # Timestampy
    grp.create_dataset('timestamps', data=timestamps_array)

    # Nazwy plików źródłowych (jako bytes dla HDF5)
    dt = h5py.special_dtype(vlen=str)
    grp.create_dataset('source_files', data=source_files, dtype=dt)

    # Metadane
    grp.attrs['num_observations'] = len(images)
    grp.attrs['image_shape'] = images_array.shape[1:]
    grp.attrs[
        'date_range'] = f"{datetime.fromtimestamp(timestamps_array.min())} - {datetime.fromtimestamp(timestamps_array.max())}"


def extract_all_mirrors(h5_path, img_path_list, extractor, mirror_ids=None):
    """Ekstrahuje wszystkie (lub wybrane) lustra i zapisuje do HDF5."""

    if mirror_ids is None:
        mirror_ids = range(200)

    with h5py.File(h5_path, 'a') as f:
        for mirror_id in tqdm(mirror_ids, desc="Ekstrakcja luster"):
            # Sprawdź czy już nie zapisano
            grp_path = f'mirrors/{mirror_id:03d}'
            if grp_path in f and 'images' in f[grp_path]:
                print(f"Mirror {mirror_id} już istnieje, pomijam")
                continue

            extract_and_save_mirror(f, mirror_id, img_path_list, extractor)


# === Funkcje pomocnicze do odczytu ===

def load_mirror_timeseries(h5_path, mirror_id):
    """Ładuje serię czasową dla jednego lustra."""
    with h5py.File(h5_path, 'r') as f:
        grp = f[f'mirrors/{mirror_id:03d}']
        images = grp['images'][:]
        timestamps = grp['timestamps'][:]
        datetimes = [datetime.fromtimestamp(ts) for ts in timestamps]
    return images, datetimes


def load_single_observation(h5_path, mirror_id, obs_index):
    """Ładuje pojedynczą obserwację (szybki dostęp dzięki chunking)."""
    with h5py.File(h5_path, 'r') as f:
        grp = f[f'mirrors/{mirror_id:03d}']
        image = grp['images'][obs_index]
        timestamp = datetime.fromtimestamp(grp['timestamps'][obs_index])
    return image, timestamp


def get_dataset_info(h5_path):
    """Wyświetla informacje o datasecie."""
    with h5py.File(h5_path, 'r') as f:
        print(f"Dataset: {h5_path}")
        print(f"Utworzono: {f.attrs.get('created', 'N/A')}")
        print(f"Liczba luster: {f.attrs.get('num_mirrors', 'N/A')}")
        print("-" * 40)

        total_obs = 0
        for mirror_id in range(200):
            grp_path = f'mirrors/{mirror_id:03d}'
            if grp_path in f and 'images' in f[grp_path]:
                n = f[grp_path].attrs.get('num_observations', 0)
                total_obs += n

        print(f"Łączna liczba obserwacji: {total_obs}")


# === MAIN ===

if __name__ == '__main__':
    BASE_DIR = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/"

    # Ścieżki
    path_to_grid = os.path.join(BASE_DIR, "data/crossings_points.pkl")
    images_dir = os.path.join(BASE_DIR, "data/webcam_useful_image/webcam_useful_images/")
    h5_output = os.path.join(BASE_DIR, "data/mirrors_dataset.h5")

    # Inicjalizacja
    extractor = MirrorExtractor(path_to_grid)
    img_list = sorted(glob.glob(os.path.join(images_dir, "*2024-05-09*.jpg")))

    print(f"Znaleziono {len(img_list)} obrazów")

    # Utwórz strukturę HDF5
    if not os.path.exists(h5_output):
        create_hdf5_dataset(h5_output, num_mirrors=200)

    # Ekstrakcja wszystkich luster
    extract_all_mirrors(h5_output, img_list, extractor)

    # Lub tylko wybranych:
    # extract_all_mirrors(h5_output, img_list, extractor, mirror_ids=[15, 150])

    # Pokaż info
    get_dataset_info(h5_output)

    # Przykład użycia
    images, dates = load_mirror_timeseries(h5_output, mirror_id=15)
    print(f"\nLustro 15: {len(images)} obserwacji")
    print(f"Zakres dat: {dates[0]} - {dates[-1]}")
    print(f"Shape pojedynczego obrazu: {images[0].shape}")
