import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from MirrorFeatureExtractor.tools import extract_polygon_region_cv2

BASE_DIR = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/"
mirror_points_path = os.path.join(BASE_DIR, "data/points_IRCam.json")
with open(mirror_points_path, 'r') as f:
    data = json.load(f)

def get_point_coords(mirror_id):
    return data['mirror_ids'][f'id_{mirror_id:03d}']

def get_mirrors_img_list(img_array):
    mirror_img_list = []
    for key in data['mirror_ids'].keys():
        points = data['mirror_ids'][key]
        crop = extract_polygon_region_cv2(img_array, np.array(points))
        mirror_img_list.append(crop)
    return mirror_img_list

def extract_features(img):
    """Extract features optimized for MAGIC telescope mirror images."""
    features = {}
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Warm bright pixels (sun/flare signature)
    hue = hsv[:,:,0].astype(float)
    sat = hsv[:,:,1].astype(float)
    val = hsv[:,:,2].astype(float)

    # ── Mask out central camera housing ──
    mask = np.ones((h, w), dtype=np.uint8) * 255
    cam_y, cam_x = int(h * 0.42), int(w * 0.65)
    cv2.circle(mask, (cam_x, cam_y), int(min(h, w) * 0.22), 0, -1)

    valid = gray[mask > 0]

    features['mean_brightness'] = np.mean(valid)
    features['std_brightness'] = np.std(valid)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features['sharpness'] = np.std(laplacian)
    warn_pixel_ratio = get_warm_bright_ratio(img)
    features["warn_pixel_ratio"] = warn_pixel_ratio

    # General bright + saturated (any color)
    bright_saturated = (val > 200) & (sat > 80) & (mask > 0)
    features['bright_saturated_frac'] = np.sum(bright_saturated) / np.sum(mask > 0)

    return features

def get_warm_bright_ratio(image):
    img = image[0:400, :]
    w, h = img.shape[0], img.shape[1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Warm bright pixels (sun/flare signature)
    hue = hsv[:,:,0].astype(float)
    sat = hsv[:,:,1].astype(float)
    val = hsv[:,:,2].astype(float)

    # Lens flare is typically warm (H < 30 or H > 160 in OpenCV) AND bright AND saturated
    warm_bright = ((hue < 25) | (hue > 160)) & (val > 180) & (sat > 60)
    return (np.sum(warm_bright)/(w*h))*100

