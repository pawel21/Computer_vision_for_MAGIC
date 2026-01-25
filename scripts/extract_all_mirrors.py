import glob
import os

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from MirrorExtractor.mirror_extractor import MirrorExtractor

from tools import extract_and_save_mirror

grid_dict = {}
grid_dict["row 1"] = list(range(4, 13))    # [4, 5, 6, 7, 8, 9, 10, 11, 12]
grid_dict["row 2"] = list(range(20, 31))   # [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
grid_dict["row 3"] = list(range(36, 49))   # [36, 37, ..., 48]
grid_dict["row 4"] = list(range(52, 67))   # [52, 53, ..., 66]
grid_dict["row 5"] = list(range(68, 85))   # [68, 69, ..., 84]
grid_dict["row 6"] = list(range(85, 102))  # [85, 86, ..., 101]
grid_dict["row 7"] = list(range(102, 119)) # [102, 103, ..., 118]
grid_dict["row 8"] = list(range(119, 136)) # [119, 120, ..., 135]
grid_dict["row 9"] = list(range(136, 153)) # [136, 137, ..., 152]
grid_dict["row 10"] = list(range(153, 170)) # [153, 154, ..., 169]
grid_dict["row 11"] = list(range(170, 187)) # [170, 171, ..., 186]
grid_dict["row 12"] = list(range(187, 204)) # [187, 188, ..., 203]
grid_dict["row 13"] = list(range(204, 221)) # [204, 205, ..., 220]
grid_dict["row 14"] = list(range(222, 237)) # [222, 223, ..., 236]
grid_dict["row 15"] = list(range(240, 253)) # [240, 241, ..., 252]
grid_dict["row 16"] = list(range(258, 269)) # [258, 259, ..., 268]
grid_dict["row 17"] = list(range(276, 285)) # [276, 277, ..., 284]

print(grid_dict)

if '__main__' == __name__:
    BASE_DIR = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/"

    path_to_grid = os.path.join(BASE_DIR, "data/crossings_points_IRCamM1T20250702_161000M.pkl")
    images_dir = os.path.join(BASE_DIR, "data/data/2025/data_test")
    #extract_all_mirrors(h5_output, img_list, extractor, mirror_ids=Mirrors_list)
    h5_output = os.path.join(BASE_DIR, "data/data/2025/mirrors_dataset_test3.h5")
    extractor = MirrorExtractor(path_to_grid)
    img_path_list = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    print(grid_dict)
    mirror_ids_list = []
    for row in grid_dict.keys():
        mirror_ids_list.extend(grid_dict[row])

    with h5py.File(h5_output, 'w') as f:
        for mirror_id in tqdm(mirror_ids_list, desc="Ekstrakcja luster"):
            extract_and_save_mirror(f, mirror_id, img_path_list, extractor)