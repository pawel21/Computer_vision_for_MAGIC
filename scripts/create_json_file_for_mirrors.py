import json
import os

from MirrorExtractor.mirror_extractor import MirrorExtractor

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

if __name__ == '__main__':
    BASE_DIR = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/"
    BASE_DIR = "/media/pgliwny/ADATA HD3303/Computer_Vision_system/"

    path_to_grid = os.path.join(BASE_DIR, "data/crossings_points_IRCamM1T20250702_161000M.pkl")
    output_path = os.path.join(BASE_DIR, "data/points_IRCam.json")

    mirror_ids_list = []
    for row in grid_dict.keys():
        mirror_ids_list.extend(grid_dict[row])

    extractor = MirrorExtractor(path_to_grid)

    points_list = []
    for i in mirror_ids_list:
        x_coords, y_coords = extractor.get_coords(i)
        points = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
        points_list.append(points)

    mirror_ids_dict = {}
    for i, p in enumerate(points_list):
        print(i, p)
        print(f"{i:03d}")
        mirror_ids_dict[f"id_{i:03d}"] = points_list[i]

    mirror_dict = {
        'calib_img': "IRCamM1T20250702_161000M",
        'mirror_ids': mirror_ids_dict
    }

    with open(output_path, mode="w") as write_file:
        json.dump(mirror_dict, write_file, indent=4)

