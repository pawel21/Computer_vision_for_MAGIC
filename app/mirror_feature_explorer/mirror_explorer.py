import numpy as np
import streamlit as st
import plotly.graph_objects as go

from PIL import Image, ImageDraw

def render_image(img):
    st.image(img, caption="MAGIC I telescope", width=400)

st.set_page_config(
    page_title="Mirror Feature Explorer",
    layout="wide",
)

st.title("Mirror Feature Explorer")

@st.cache_data
def load_data(path):
    data = np.load(path, allow_pickle=True)
    feat_array = data['feature_matrix']
    feat_names = data['feature_names']
    return feat_array, feat_names

@st.cache_data
def load_img(path):
    img = Image.open(path)
    return img

fm, feat_names = load_data("../../data/feature_matrix_1.npz")
img = load_img("data/IRCamM1T20250915_163000M.jpg")

feature_names = ['lbp_entropy', 'glcm_contrast', 'glcm_dissimilarity',
                 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
                 "sobel_mean", "soble_max",
                 "laplacian_mean", "laplacian_std", "edge_density"]

feat_names_dict = {}
i = 0
for name in feature_names:
    feat_names_dict[name] = i
    i += 1

print(feat_names_dict)

T, N_mirrors, N_feats = fm.shape
st.write(f"Dane: {T} zdjęć, {N_mirrors} luster, {N_feats} features")


with st.sidebar:
    st.header("Settings")

    selected_mirror = st.multiselect(
        "Mirror (ID 0-248)",
        options=list(range(0, N_mirrors)),
        default=[1, 2, 3],
        help="Choose one or more mirror for plot"
    )

    selected_feature = st.selectbox(
        "Choose feature",
        options=feat_names,
        help="Choose feature for plot"
    )

    st.divider()
    st.write(f"Mirror: {selected_mirror}, Feature: {selected_feature}, feat id: {feat_names_dict[selected_feature]}")

# Wykres
st.subheader(f"Feature: {selected_feature}")

render_image(img)

fig = go.Figure()
feat_id = feat_names_dict[selected_feature]

for mirror_id in selected_mirror:
    series = fm[:, mirror_id, feat_id]

    fig.add_trace(go.Scatter(
        x=list(range(T)),
        y=series,
        mode = "lines+markers",
        name = mirror_id,
        marker=dict(size=4),
    ))

fig.update_layout(
    xaxis_title="ID",
    yaxis_title=selected_feature,
    hovermode="x unified",
    height=450,
)

st.plotly_chart(fig, use_container_width = True)