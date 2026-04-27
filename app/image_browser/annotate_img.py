import gradio as gr
from pathlib import Path
from PIL import Image, ImageDraw

from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import json

BASE_DIR = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/"
# BASE_DIR = "/media/pgliwny/ADATA HD3303/Computer_Vision_system/"
mirror_points_path = os.path.join(BASE_DIR, "data/points_IRCam.json")
#img_path = os.path.join(BASE_DIR, "data/data/2025/12/15/IRCamM1T20251215_122547M.jpg")

with open(mirror_points_path, 'r') as f:
    data = json.load(f)

def get_point_coords(mirror_id):
    return data['mirror_ids'][f'id_{mirror_id:03d}']

def show_mirror_ids(mirror_ids):
    print(mirror_ids)

def mark_mirror_on_img(img_path, input_text):
    print(input_text)
    img = PILImage.open(img_path)
    mirror_ids_list = [int(x.strip()) for x in input_text.split(",") if x.strip()]
    print(f"Mirror ids: {mirror_ids_list}")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    for mirror_id in mirror_ids_list:
        points = get_point_coords(mirror_id)
        # === Polygon na podstawie 4 współrzędnych ===
        polygon = patches.Polygon(
            points,
            closed=True,
            linewidth=2,
            edgecolor='red',
            facecolor='red',
            alpha=0.3  # przezroczystość wypełnienia
        )
        ax.add_patch(polygon)

    plt.tight_layout()
    return fig

def load_calib_file(calib_file):
    if calib_file is None:
        return None, "Nie wczytano pliku"
    with open(calib_file, "r") as f:
        calib_data = json.load(f)
    return calib_data, f"✅ Wczytano plik: {len(calib_data)} kluczy"

with gr.Blocks() as demo:
    calib_state = gr.State(None) # tutaj trzymamy dane kalibracyjne
    with gr.Row():
        input_img = gr.Image(type="filepath", label="img 1", height=500)
        output_plot = gr.Plot(label="Img")
    with gr.Row():
        calib_file = gr.File(label="Calib file JSON", file_types=[".json"])
        calib_status = gr.Textbox(label="Calib status", interactive=False)
    calib_file.change(
        load_calib_file,
        inputs=[calib_file],
        outputs=[calib_state, calib_status]
    )
    input_mirror_ids = gr.Textbox()
    show_btn = gr.Button("show")
    textbox = gr.Textbox(label="Raport", lines=15)
    print(input_img)
    show_btn.click(mark_mirror_on_img, inputs=[input_img, input_mirror_ids], outputs=[output_plot])

demo.launch()