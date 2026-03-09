import gradio as gr
from pathlib import Path
from PIL import Image, ImageDraw

from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import json



class ImageBrowser:
    def __init__(self):
        self.images = []
        self.current = 0

    def load(self, folder):
        self.images = sorted(Path(folder).rglob("*.jpg"))
        self.current = 0
        return self.show()


    def show(self):
        if not self.images:
            return None, "Brak"
        img = Image.open(self.images[self.current])

        draw = ImageDraw.Draw(img) # tool to draw

        draw.rectangle(
            [(200,300), (300,400)],
            outline="red",
            width=30
        )
        points = [(245, 85), (315, 76), (312, 156), (241, 165), (245, 85)]
        draw.polygon(points, outline="red", width=10)
        info = f"{self.current + 1} / {len(self.images)}"
        return img, info

    def next(self):
        if self.current < len(self.images) - 1:
            self.current += 1
        return self.show()

BASE_DIR = "/home/pgliwny/Praca/Computer_vision_for_MAGIC/"
mirror_points_path = os.path.join(BASE_DIR, "data/points_IRCam.json")
img_path = os.path.join(BASE_DIR, "data/data/2025/12/15/IRCamM1T20251215_122547M.jpg")

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

browser = ImageBrowser()

with gr.Blocks() as demo:
    folder = gr.Textbox(value=".")
    load_btn = gr.Button("Load") # ../../data/data/2025/06/27/
    img = gr.Image(height=800, width=800)
    info = gr.Textbox()
    next_btn = gr.Button("Next Image")

    load_btn.click(browser.load, inputs=folder, outputs=[img, info])
    next_btn.click(browser.next, outputs=[img, info])

with demo.route("Mirror segmentation", "mirror"):
    with gr.Row():
        input_img = gr.Image(type="filepath", label="img 1", height=500)
        output_plot = gr.Plot(label="Img")
    input_mirror_ids = gr.Textbox()
    show_btn = gr.Button("show")
    textbox = gr.Textbox(label="Raport", lines=15)
    print(input_img)
    show_btn.click(mark_mirror_on_img, inputs=[input_img, input_mirror_ids], outputs=[output_plot])


demo.launch()