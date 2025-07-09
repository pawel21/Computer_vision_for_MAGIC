import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image

def get_coords(state, indices, m_points):
    i, j = indices[state['idx']]
    x_coords = [
        m_points[i, j]['x'],
        m_points[i, j + 1]['x'],
        m_points[i + 1, j + 1]['x'],
        m_points[i + 1, j]['x'],
        m_points[i, j]['x']
    ]
    y_coords = [
        m_points[i, j]['y'],
        m_points[i, j + 1]['y'],
        m_points[i + 1, j + 1]['y'],
        m_points[i + 1, j]['y'],
        m_points[i, j]['y']
    ]
    return x_coords, y_coords

def get_matrix_points(crossing_points_path='/media/pgliwny/ADATA HD330/Computer_Vision_system/data/crossings_points.pkl'):
    with open(crossing_points_path, 'rb') as f:
        crossing_points = pickle.load(f)

    dt = np.dtype([('x', 'i4'), ('y', 'i4')])
    m_points = np.zeros((18, 18), dtype=dt)
    i = 0
    j = 0
    for p in crossing_points:
        m_points[i, j] = p
        j += 1
        if j % 18 == 0:
            j = 0
            i += 1
    return m_points

def add_box_around_mirror(img_np, list_idx, m_points):
    rows, cols = m_points.shape
    indices = [(i, j) for i in range(rows - 1) for j in range(cols - 1)]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.imshow(img_np)
    for idx in list_idx:
        state = {'idx': idx}
        x_coords, y_coords = get_coords(state, indices, m_points)
        ax1.plot(x_coords, y_coords, 'r-', lw=0.5)
        ax1.scatter(x_coords[:-1], y_coords[:-1], c='cyan', s=10)
    ax1.axis('off')
    return fig



st.title("Mirror Bounding Box Viewer")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
indces_str = st.text_input("Enter mirror indices (coma-separated)", "0, 1, 2")
print(indces_str)

if uploaded_image is not None and indces_str:
	try:
		list_idx = [int(i.strip()) for i in indces_str.split(',')]
		image = Image.open(uploaded_image).convert("RGB")
		img_np = np.array(image)

		m_points = get_matrix_points()
		fig = add_box_around_mirror(img_np, list_idx, m_points)
		#fig, ax = plt.subplots()
		#ax.imshow(img_np)
		#ax.set_title(indces_str)
		
		st.pyplot(fig)
		
	except Exception as e:
		st.error(f"An error occurred: {e}")