import gradio as gr
from pathlib import Path
from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

from MirrorFeatureExtractor.comparator import get_diff_all_features, get_texture_feat, find_outlier_mirrors
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

def compare_images(img1, img2):
    """
    Porównuje dwa obrazy i zaznacza przykładowe punkty
    """
    if img1 is None or img2 is None:
        return None

    # Konwertuj do numpy array
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    mirror_img_list_1 = get_mirrors_img_list(arr1)
    mirror_img_list_2 = get_mirrors_img_list(arr2)
    output_diff = get_diff_all_features(mirror_img_list_1, mirror_img_list_2)
    o = find_outlier_mirrors(output_diff, n_top=10)

    mirror_list = []
    for i in range(0, 10):
        mirror_list.append(o[i]['mirror_idx'])

    # Stwórz wykres z trzema panelami
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    # Panel 1: Pierwszy obraz z punktami
    axes[0].imshow(arr1)

    axes[0].set_title('Obraz 1 - punkty kluczowe', fontsize=14)
    axes[0].axis('off')


    # Panel 2: Drugi obraz z punktami
    axes[1].imshow(arr2)

    axes[1].set_title('Obraz 2 - punkty kluczowe', fontsize=14)
    axes[1].axis('off')

    for m_id in mirror_list:
        points = get_point_coords(m_id)
        polygon1 = patches.Polygon(points,
                              linewidth=5,
                              edgecolor='g',
                              facecolor='none')
        axes[0].add_patch(polygon1)

        polygon2 = patches.Polygon(points,
                                   linewidth=5,
                                   edgecolor='g',
                                   facecolor='none')
        axes[1].add_patch(polygon2)



    plt.tight_layout()

    return fig

with gr.Blocks(title="Image comparision") as demo:
    gr.Markdown("# MAGIC image comparator")
    with gr.Row():
        input1 = gr.Image(type="pil", label="img 1", height=600)
        input2 = gr.Image(type="pil", label="img 2", height=600)

    compare_btn = gr.Button("Compare", variant="primary", size="lg")

    output_plot = gr.Plot(label="result")

    compare_btn.click(
        fn=compare_images,
        inputs=[input1, input2],
        outputs=output_plot
    )

if __name__ == '__main__':
    demo.launch()