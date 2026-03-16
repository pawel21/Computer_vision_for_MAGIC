import numpy as np
import streamlit as st
import plotly.graph_objects as go

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

fm, feat_names = load_data("data/feature_matrix_1.npz")

T, N_mirrors, N_feats = fm.shape
st.write(f"Dane: {T} zdjęć, {N_mirrors} luster, {N_feats} features")
st.write(feat_names)

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
    st.write(f"Mirror: {selected_mirror}, Feature: {selected_feature}")

# Wykres
st.subheader(f"Feature: {selected_feature}")

fig = go.Figure()

for mirror_id in selected_mirror:
    series = fm[:, mirror_id, 0]

    fig.add_trace(go.Scatter(
        x=list(range(T)),
        y=fm[:, mirror_id, 1],
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