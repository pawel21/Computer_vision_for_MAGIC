import cv2
import os
import glob
import shutil
from pathlib import Path

from tools import extract_features

DIR_PATH = None
OUTPUT_DIR_PATH = "/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/MAGIC/images_for_analysis"



root = Path("/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/MAGIC/IRCam/IRCamera/2025/10")
good_img_path_list = []
for day_dir in sorted(root.iterdir()):
    if day_dir.is_dir():
        print(day_dir.name)
        # all images from this day
        for photo_path in day_dir.glob("*T2*.jpg"):
            img = cv2.imread(str(photo_path))
            feat = extract_features(img)
            if feat['mean_brightness'] > 100 and feat["bright_saturated_frac"] < 0.025 and feat['sharpness'] > 25:
                fname = str(photo_path).split("/")[-1]
                new_path = os.path.join(OUTPUT_DIR_PATH, fname)
                shutil.copy(photo_path, new_path)





