import gradio as gr
from pathlib import Path
from PIL import Image, ImageDraw


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

browser = ImageBrowser()

with gr.Blocks() as demo:
    folder = gr.Textbox(value=".")
    load_btn = gr.Button("Load") # ../../data/data/2025/06/27/
    img = gr.Image(height=800, width=800)
    info = gr.Textbox()
    next_btn = gr.Button("Next Image")

    load_btn.click(browser.load, inputs=folder, outputs=[img, info])
    next_btn.click(browser.next, outputs=[img, info])

demo.launch()