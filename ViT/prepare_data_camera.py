import numpy as np
from PIL import Image
from pathlib import Path

background_img_path = Path("background/")
camera_background_img_path = Path("camera_background/")

for path in background_img_path.glob("*.jpg"):
    print(path)
    img = Image.open(path).convert('RGB')
    img_array = np.array(img)[850:1350, :400, :]
    new_img = Image.fromarray(img_array).convert('RGB')
    new_img.save(camera_background_img_path/path.name)

test_img_path = Path("test/")
camera_test_img_path = Path("camera_test/")

for path in test_img_path.glob("*.jpg"):
    print(path)
    img = Image.open(path).convert('RGB')
    img_array = np.array(img)[850:1350, :400, :]
    new_img = Image.fromarray(img_array).convert('RGB')
    new_img.save(camera_test_img_path/path.name)
