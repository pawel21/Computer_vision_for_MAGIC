import gradio as gr
from pathlib import Path
from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

from MirrorFeatureExtractor.comparator import (
    get_diff_all_features,
    get_texture_feat,
    find_outlier_mirrors,
    get_outlier_mirrors_report
 )
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

def compare_images(img1_path, img2_path):
    """
    Porównuje dwa obrazy i zaznacza przykładowe punkty
    """

    filename1 = os.path.basename(img1_path)  # tylko nazwa pliku
    filename2 = os.path.basename(img2_path)

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    # Konwertuj do numpy array
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    mirror_img_list_1 = get_mirrors_img_list(arr1)
    mirror_img_list_2 = get_mirrors_img_list(arr2)
    output_diff = get_diff_all_features(mirror_img_list_1, mirror_img_list_2)
    text_out = get_outlier_mirrors_report(output_diff, n_top=5)
    text_out += "\n" + filename1
    text_out += "\n" + filename2

    distances = [fd['distance'] for fd in output_diff]

    mirror_list = list(np.where(np.array(distances) > (np.mean(distances) + 1.5*np.std(distances)))[0])
    print(mirror_list)

    # Stwórz wykres z trzema panelami
    fig1, axes = plt.subplots(1, 2, figsize=(12, 7))

    # Panel 1: Pierwszy obraz z punktami
    axes[0].imshow(arr1)

    axes[0].set_title('Obraz 1', fontsize=14)
    axes[0].axis('off')


    # Panel 2: Drugi obraz z punktami
    axes[1].imshow(arr2)

    axes[1].set_title('Obraz 2', fontsize=14)
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

    fig2, ax = plt.subplots(1, 1, figsize=(7, 7))

    distances = [fd['distance'] for fd in output_diff]
    mirror_ids = [fd['mirror_idx'] for fd in output_diff]

    ax.bar(mirror_ids, distances)
    ax.axhline(np.mean(distances) + 1.5 * np.std(distances), color='r',
                linestyle='--', label='Mean + 1.5σ')
    ax.set_xlabel('Mirror ID')
    ax.set_ylabel('Distance (normalized)')
    ax.set_title('Odległość cech tekstury między zdjęciami A i B')
    plt.legend()

    return fig1, fig2, text_out

with gr.Blocks(title="Image comparision") as demo:
    gr.Markdown("# MAGIC image comparator")
    with gr.Row():
        input1 = gr.Image(type="filepath", label="img 1", height=600)
        input2 = gr.Image(type="filepath", label="img 2", height=600)

    compare_btn = gr.Button("Compare", variant="primary", size="lg")

    output_plot = gr.Plot(label="result")
    output_plot_2 = gr.Plot(label="result 2")
    textbox = gr.Textbox(label="Raport", lines=15)

    compare_btn.click(
        fn=compare_images,
        inputs=[input1, input2],
        outputs=[output_plot, output_plot_2, textbox]
    )

if __name__ == '__main__':
    demo.launch()