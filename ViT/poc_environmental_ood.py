import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt

# ---------- KONFIGURACJA ----------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 518  # DINOv2 lubi wielokrotności 14
PATCH_SIZE = 14
GRID = IMG_SIZE // PATCH_SIZE  # 37
N_CLUSTERS = 30
THRESHOLD = 0.7  # będziesz tuningował

# ---------- MODEL ----------
print("Ładowanie DINOv2...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval().to(DEVICE)

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------- EKSTRAKCJA CECH ----------
def extract_patches(img_path):
    """Zwraca patch embeddings: (GRID*GRID, feature_dim)"""
    img = Image.open(img_path).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = model.forward_features(x)
    return feats['x_norm_patchtokens'].squeeze(0).cpu().numpy()


# ---------- BUDOWA BANKU PROTOTYPÓW ----------
def build_prototypes(background_image_paths):
    print(f"Ekstrakcja cech z {len(background_image_paths)} obrazów tła...")
    all_patches = []
    for p in background_image_paths:
        patches = extract_patches(p)
        all_patches.append(patches)
    all_patches = np.vstack(all_patches)
    print(f"Łącznie patchy: {all_patches.shape}")

    print(f"K-means z k={N_CLUSTERS}...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    kmeans.fit(all_patches)
    return kmeans.cluster_centers_  # (N_CLUSTERS, dim)


# ---------- DETEKCJA OOD ----------
def detect_ood(img_path, prototypes, threshold=THRESHOLD):
    patches = extract_patches(img_path)  # (1369, 384)

    # normalizacja L2 dla cosine similarity
    patches_n = patches / (np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8)
    protos_n = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)

    # cosine sim do najbliższego prototypu
    sims = patches_n @ protos_n.T  # (1369, N_CLUSTERS)
    max_sim = sims.max(axis=1)  # (1369,)

    # reshape do siatki 37x37
    sim_map = max_sim.reshape(GRID, GRID)
    ood_mask = (sim_map < threshold).astype(np.uint8)

    return sim_map, ood_mask


# ---------- WIZUALIZACJA ----------
def visualize(img_path, sim_map, ood_mask, save_path=None):
    img = np.array(Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))

    # upsample do rozmiaru obrazu
    sim_up = cv2.resize(sim_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    mask_up = cv2.resize(ood_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # morfologia - usuń pojedyncze patche
    kernel = np.ones((10, 10), np.uint8)
    mask_clean = cv2.morphologyEx(mask_up, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img);
    axes[0].set_title('Original');
    axes[0].axis('off')
    axes[1].imshow(sim_up, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_title('Max similarity (red=OOD)');
    axes[1].axis('off')
    axes[2].imshow(mask_clean, cmap='gray')
    axes[2].set_title(f'OOD mask (thr={THRESHOLD})');
    axes[2].axis('off')

    # overlay
    overlay = img.copy()
    overlay[mask_clean > 0] = [255, 0, 0]
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    axes[3].imshow(blended);
    axes[3].set_title('Overlay');
    axes[3].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


# ---------- URUCHOMIENIE ----------
if __name__ == '__main__':
    # ŚCIEŻKI - tu wstaw swoje
    background_dir = Path('data/camera_background')  # 10 obrazów czystego tła
    test_dir = Path('data/camera_test')  # obrazy z drabiną/ludźmi
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    background_imgs = sorted(background_dir.glob('*.jpg'))
    test_imgs = sorted(test_dir.glob('*.jpg'))

    # 1. Budowa banku
    prototypes = build_prototypes(background_imgs)
    np.save('prototypes.npy', prototypes)

    # 2. Test na obrazach z anomaliami
    for img_path in test_imgs:
        print(f"\nDetekcja: {img_path.name}")
        sim_map, ood_mask = detect_ood(img_path, prototypes)
        print(f"  min sim: {sim_map.min():.3f}, max sim: {sim_map.max():.3f}")
        print(f"  patche OOD: {ood_mask.sum()}/{GRID * GRID}")

        visualize(img_path, sim_map, ood_mask,
                  save_path=output_dir / f'{img_path.stem}_ood.png')