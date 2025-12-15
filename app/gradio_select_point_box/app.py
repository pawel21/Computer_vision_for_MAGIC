import gradio as gr
import numpy as np
from PIL import Image
import json
import cv2

annotations_data = {
    "points": [],
    "boxes": []
}

def add_point(image, evt: gr.SelectData):
    """Add a point when user clicks on image"""
    if image is None:
        return None, "No image loaded"

    x, y = evt.index[0], evt.index[1]
    annotations_data["points"].append({"x": x, "y": y})

    # Rysuj punkty na obrazie
    img_with_points = np.array(image.copy())

    # Rysuj wszystkie punkty
    for point in annotations_data["points"]:
        px, py = point["x"], point["y"]
        # Czerwone kółko
        cv2.circle(img_with_points, (px, py), 5, (255, 0, 0), -1)
        # Czarna obwódka
        cv2.circle(img_with_points, (px, py), 6, (0, 255, 0), 2)

    result_image = Image.fromarray(img_with_points)

    # Create output
    output = f"Points: {len(annotations_data['points'])}\n"
    output += f"Last point: ({x}, {y})\n\n"
    output += "All points:\n"
    output += json.dumps(annotations_data["points"], indent=2)

    return result_image, output


def clear_annotations():
    """Clear all annotations"""
    annotations_data["points"] = []
    annotations_data["boxes"] = []
    return None, "Annotations cleared"


def save_annotations():
    """Return annotations as JSON string"""
    return json.dumps(annotations_data, indent=2)


# Create interface
with gr.Blocks() as demo:
    gr.Markdown("# Point Selector")

    with gr.Row():
        with gr.Column():
            image = gr.Image(
                label="Click on image to add points",
                type="pil"
            )

            with gr.Row():
                clear_btn = gr.Button("Clear All")
                save_btn = gr.Button("Get JSON", variant="primary")

        with gr.Column():
            output = gr.Textbox(
                label="Coordinates",
                lines=20,
                interactive=False
            )

    json_output = gr.Textbox(
        label="JSON Output (copy this)",
        lines=5,
        interactive=False
    )

    # Event handlers
    image.select(add_point, inputs=[image], outputs=[image, output])
    clear_btn.click(clear_annotations, outputs=[json_output])
    save_btn.click(save_annotations, outputs=[json_output])

    gr.Markdown("""
        ### How to use:
        1. Upload an image
        2. Click on the image to add points
        3. Click "Get JSON" to export coordinates
        4. Use these coordinates with SAM

        ### Example SAM usage:
        ```python
        import json

        # Load your annotations
        data = json.loads(annotations_json)
        points = [[p['x'], p['y']] for p in data['points']]

        # Use with SAM
        masks, scores, logits = predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array([1] * len(points)),  # 1 = positive point
            multimask_output=True,
        )
        ```
        """)

if __name__ == '__main__':
    demo.launch(share=False)