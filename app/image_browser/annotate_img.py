import gradio as gr
from pathlib import Path
from PIL import Image, ImageDraw

from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import json


def mark_mirror_on_img(img_path, input_text, calib_data):
    img = PILImage.open(img_path)
    mirror_ids_list = [int(x.strip()) for x in input_text.split(",") if x.strip()]
    print(f"Mirror ids: {mirror_ids_list}")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    for mirror_id in mirror_ids_list:
        points = calib_data['mirror_ids'][f'id_{mirror_id:03d}']
        polygon = patches.Polygon(
            points,
            closed=True,
            linewidth=2,
            edgecolor='red',
            facecolor='red',
            alpha=0.3
        )
        ax.add_patch(polygon)

    report = {
        "file": os.path.basename(img_path) if img_path else None,
        "marked_mirrors": [int(x.strip()) for x in input_text.split(",") if x.strip()]
    }

    plt.tight_layout()
    return fig, json.dumps(report, indent=2, ensure_ascii=False)

def load_calib_file(calib_file):
    if calib_file is None:
        return None, "No load file"
    with open(calib_file, "r") as f:
        calib_data = json.load(f)
    return calib_data, f"Load file: {len(calib_data)} keys"

with gr.Blocks() as demo:
    calib_state = gr.State(None) # Here hold calib file
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
    report_box = gr.Textbox(label="Raport", lines=15)
    show_btn.click(mark_mirror_on_img,
                   inputs=[input_img, input_mirror_ids, calib_state],
                   outputs=[output_plot, report_box])

demo.launch()