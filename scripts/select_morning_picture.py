import glob
import cv2
import os
import glob
import shutil
import re
from pathlib import Path

from tools import extract_features

def get_hour(path):
    match = re.search(r'(\d{8})_(\d{6})M', path)
    if match:
        time_str = match.group(2)  # "152000"
        return int(time_str[:2])   # 15
    return None

image_paths = sorted(glob.glob("/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/data/images_for_analysis/*.jpg"))
morning_img_paths = [p for p in image_paths if 7 <= get_hour(p) < 10]

OUTPUT_DIR_PATH = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/data/images_for_analysis/morning"
root = Path("/home/pgliwny/Praca/Computer_vision_for_MAGIC/data/data/2025/12/")

good_img_path_list = []

for day_dir in sorted(root.iterdir()):
    if day_dir.is_dir():
        print(day_dir.name)
        # all images from this day
        for photo_path in day_dir.glob("*T2*.jpg"):
            hour = get_hour(str(photo_path))
            if hour > 7 and hour < 9:
                img = cv2.imread(str(photo_path))
                if img is None:
                    print(f"Can not read {photo_path}")
                else:
                    feat = extract_features(img)
                    if feat is None:
                        print(f"Can not extract features for {photo_path}")
                    elif feat['mean_brightness'] > 100 and feat['mean_brightness'] < 145 and feat['sharpness'] > 25:
                        fname = str(photo_path).split("/")[-1]
                        new_path = os.path.join(OUTPUT_DIR_PATH, fname)
                        shutil.copy(photo_path, new_path)