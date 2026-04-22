import glob
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap

# --- 1. Model ---
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()

transform = T.Compose([
    T.Resize((518, 518)),  # wielokrotność 14
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# --- 2. Ekstrakcja embeddingów ---
def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(x)  # shape: (1, 384)
    return emb.squeeze().numpy()

image_paths = glob.glob("/media/pgliwny/ADATA HD3303/Computer_Vision_system/data/MAGIC/images_test/*")

embeddings = np.array([get_embedding(p) for p in image_paths])
# shape: (N, 384)

# --- 3. Redukcja wymiarowości ---
# Przy małym N: PCA do 2D wystarczy
# Przy dużym N (setki obrazów): UMAP lepszy
reducer = umap.UMAP(n_components=2, random_state=42)
emb_2d = reducer.fit_transform(embeddings)

# --- 4. Klastrowanie ---
# KMeans (gdy znasz liczbę klastrów)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(embeddings)

# lub DBSCAN (gdy nie znasz)
# dbscan = DBSCAN(eps=0.5, min_samples=2)
# labels = dbscan.fit_predict(StandardScaler().fit_transform(embeddings))

# --- 5. Wizualizacja ---
plt.figure(figsize=(8, 6))
scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', s=100)
for i, path in enumerate(image_paths):
    plt.annotate(path.split("_")[-1][:6], (emb_2d[i, 0], emb_2d[i, 1]),
                 fontsize=8, ha='right')
plt.colorbar(scatter, label='Klaster')
plt.title('DINOv2 embeddings – klastrowanie obrazów IRCam')
plt.tight_layout()
plt.savefig('output/clusters.png', dpi=150)