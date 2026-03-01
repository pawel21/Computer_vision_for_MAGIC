import cv2
import os
import glob
import shutil
from pathlib import Path

from tools import extract_features


OUTPUT_DIR_PATH = "/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/MAGIC/images_for_analysis"
root = Path("/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/MAGIC/IRCam/IRCamera/2025/10")

MY_COMP_OUTPUT_DIR_PATH = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/data/images_for_analysis"
my_comp_root = Path("/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/data/2025/10/")

good_img_path_list = []

for day_dir in sorted(my_comp_root.iterdir()):
    if day_dir.is_dir():
        print(day_dir.name)
        # all images from this day
        for photo_path in day_dir.glob("*T2*.jpg"):
            img = cv2.imread(str(photo_path))
            if img is None:
                print(f"Can not read {photo_path}")
            else:
                feat = extract_features(img)
                if feat is None:
                    print(f"Can not extract features for {photo_path}")
                elif feat['mean_brightness'] > 100 and feat["bright_saturated_frac"] < 0.025 and feat['sharpness'] > 25:
                    fname = str(photo_path).split("/")[-1]
                    new_path = os.path.join(MY_COMP_OUTPUT_DIR_PATH, fname)
                    shutil.copy(photo_path, new_path)





