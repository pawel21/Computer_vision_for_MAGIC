# main script

import os
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor

BASE_DIR = "/media/pgliwny/ADATA HD3303/Computer_Vision_system/"
ROOT = Path(BASE_DIR) / "data/MAGIC/images_for_analysis"
MIRROR_POINTS_JSON = Path(BASE_DIR) / "data/points_IRCam.json"

mirror_extractor = SimpleMirrorExtractor(str(MIRROR_POINTS_JSON))
print(mirror_extractor)

for photo_path in sorted(ROOT.glob("*20251030*.jpg")):
    print(photo_path)
    img_bgr = cv2.imread(str(photo_path))  # Path → str required by OpenCV
    if img_bgr is None:
        print(f"  Warning: could not read {photo_path}, skipping.")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mirror_crop = mirror_extractor.extract_mirror(img_rgb, mirror_id=150)

    plt.figure()
    plt.title(photo_path.name)
    plt.imshow(mirror_crop, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()