import cv2
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def mark_mirrors_on_img(img_path, points_list, color, ax):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    ax.imshow(img_rgb)

    for points in points_list:
        # === Polygon na podstawie 4 współrzędnych ===
        polygon = patches.Polygon(
                points,
                closed=True,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.3  # przezroczystość wypełnienia
            )
        ax.add_patch(polygon)
