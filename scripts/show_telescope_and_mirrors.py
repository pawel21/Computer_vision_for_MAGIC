
from glob import glob
import plotly.graph_objects as go
from PIL import Image

def create_interactive_image_viewer(img_paths):
    """
    Create an interactive image viewer with Plotly's built-in dropdown
    No ipywidgets required!
    """
    # Load all images
    images = [Image.open(path) for path in img_paths]
    
    # Create figure
    fig = go.Figure()
    
    # Add all images as layout images
    for i, (img, path) in enumerate(zip(images, img_paths)):
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=img.size[1],
                sizex=img.size[0],
                sizey=img.size[1],
                sizing="stretch",
                layer="below",
                visible=(i == 0)
            )
        )
    
    print(images)
    # Get max dimensions
    max_width = max(img.size[0] for img in images)
    max_height = max(img.size[1] for img in images)
    
    # Create dropdown buttons
    buttons = []
    for i, path in enumerate(img_paths):
        visible = [j == i for j in range(len(img_paths))]
        
        buttons.append(
            dict(
                label=f"Image {i+1}",
                method="relayout",
                args=[
                    {
                        **{f"images[{j}].visible": visible[j] for j in range(len(img_paths))},
                        "title.text": f"Image {i+1}: {path.split('/')[-1]}"
                    }
                ]
            )
        )
    
    # Update layout with dropdown
    fig.update_layout(
        title=f"Image 1: {img_paths[0].split('/')[-1]}",
        width=900,
        height=700,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, max_width]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, max_height], scaleanchor="x"),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    return fig


img_list = glob("../data/webcam_useful_image/webcam_useful_images/*.jpg")
img_list[:10]

# Use it:
fig = create_interactive_image_viewer(img_list[:10])
fig.show()
