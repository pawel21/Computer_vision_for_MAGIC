import cv2
from skimage import feature
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA
from scipy.stats import entropy
from skimage.filters import sobel
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection, LineCollection

def extract_polygon_region_cv2(img_array, pts):
    img = img_array
    pts = pts.reshape((-1, 1, 2))

    # Create a mask with the same shape as the image (1 channel)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=255)

    # Apply mask to image
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Find bounding rectangle and crop
    x, y, w, h = cv2.boundingRect(pts)
    cropped = masked_img[y:y + h, x:x + w]
    cropped_gray = rgb_to_gray(cropped)
    return cropped_gray


RGB_TO_GRAY = np.array([0.299, 0.587, 0.114])

def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """Konwersja RGB do skali szarośći metodą weighted average"""
    return np.dot(img_rgb[..., :3], RGB_TO_GRAY).astype(np.uint8)


def compute_advanced_features(mirror_list):
    """
    Oblicza zaawansowane cechy niezmiennicze na oświetlenie
    """
    features_list = []
    
    for mirror in mirror_list:
        feat = {}
        
        # === 1. CECHY KOLORYSTYCZNE (niezmiennicze na jasność) ===
        
        # Normalizowane proporcje RGB (suma = 1)
        mean_rgb = np.mean(mirror, axis=(0, 1))
        rgb_sum = np.sum(mean_rgb)
        if rgb_sum > 0:
            feat['r_ratio'] = mean_rgb[0] / rgb_sum
            feat['g_ratio'] = mean_rgb[1] / rgb_sum
            feat['b_ratio'] = mean_rgb[2] / rgb_sum
        
        # Stosunek kolorów (niezależny od jasności)
        feat['rg_ratio'] = mean_rgb[0] / (mean_rgb[1] + 1e-6)
        feat['rb_ratio'] = mean_rgb[0] / (mean_rgb[2] + 1e-6)
        feat['gb_ratio'] = mean_rgb[1] / (mean_rgb[2] + 1e-6)
        
        # HSV - Hue jest niezmiennicza na oświetlenie!
        hsv = cv2.cvtColor(mirror, cv2.COLOR_RGB2HSV)
        feat['hue_mean'] = np.mean(hsv[:, :, 0])
        feat['hue_std'] = np.std(hsv[:, :, 0])
        feat['saturation_mean'] = np.mean(hsv[:, :, 1])
        feat['saturation_std'] = np.std(hsv[:, :, 1])
        
        # Histogram Hue (dominujący kolor)
        hue_hist, _ = np.histogram(hsv[:, :, 0], bins=16, range=(0, 180))
        hue_hist = hue_hist / (hue_hist.sum() + 1e-6)
        feat['hue_entropy'] = -np.sum(hue_hist * np.log2(hue_hist + 1e-6))
        feat['hue_peak'] = np.argmax(hue_hist)
        
        # === 2. TEKSTURA (niezmiennicza na oświetlenie) ===
        
        gray = cv2.cvtColor(mirror, cv2.COLOR_RGB2GRAY)
        
        # LBP (Local Binary Patterns) - bardzo odporna na oświetlenie
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
        
        for i, val in enumerate(lbp_hist[:5]):
            feat[f'lbp_bin_{i}'] = val
        
        # GLCM (Gray Level Co-occurrence Matrix)
        # Normalizuj gray do 0-63 dla szybszości
        gray_norm = (gray / 4).astype(np.uint8)
        glcm = feature.graycomatrix(gray_norm, distances=[1], angles=[0], 
                                     levels=64, symmetric=True, normed=True)
        
        feat['glcm_contrast'] = feature.graycoprops(glcm, 'contrast')[0, 0]
        feat['glcm_dissimilarity'] = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
        feat['glcm_homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
        feat['glcm_energy'] = feature.graycoprops(glcm, 'energy')[0, 0]
        feat['glcm_correlation'] = feature.graycoprops(glcm, 'correlation')[0, 0]
        
        # === 3. KSZTAŁT ROZKŁADU ===
        
        # Skewness i Kurtosis (niezmiennicze na przesunięcie)
        for i, color in enumerate(['r', 'g', 'b']):
            channel = mirror[:, :, i].flatten()
            feat[f'{color}_skewness'] = skew(channel)
            feat[f'{color}_kurtosis'] = kurtosis(channel)
        
        # === 4. CZĘSTOTLIWOŚĆ (FFT) ===
        
        # FFT - wzory częstotliwościowe są niezmiennicze na oświetlenie
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Energia w pasmach częstotliwości
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Niskie częstotliwości (środek)
        low_freq = magnitude[center_h-2:center_h+2, center_w-2:center_w+2]
        feat['fft_low_freq'] = np.mean(low_freq)
        
        # Wysokie częstotliwości (rogi)
        high_freq = np.concatenate([
            magnitude[:2, :].flatten(),
            magnitude[-2:, :].flatten(),
            magnitude[:, :2].flatten(),
            magnitude[:, -2:].flatten()
        ])
        feat['fft_high_freq'] = np.mean(high_freq)
        
        # Stosunek wysokie/niskie
        feat['fft_ratio'] = feat['fft_high_freq'] / (feat['fft_low_freq'] + 1e-6)
        
        # === 5. EDGE DENSITY (gęstość krawędzi) ===
        
        edges = cv2.Canny(gray, 50, 150)
        feat['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Kierunek krawędzi (gradient)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_dir = np.arctan2(sobely, sobelx)
        feat['gradient_dir_std'] = np.std(gradient_dir)
        
        # === 6. SPATIAL MOMENTS ===
        
        # Momenty obrazu (niezmiennicze na oświetlenie gdy znormalizowane)
        moments = cv2.moments(gray)
        if moments['m00'] > 0:
            feat['hu_moment_1'] = moments['mu20'] / (moments['m00'] ** 2 + 1e-6)
            feat['hu_moment_2'] = moments['mu02'] / (moments['m00'] ** 2 + 1e-6)
        
        # === 7. UNIFORMITY I SMOOTHNESS ===
        
        # Uniformity (jak równomierny jest obraz)
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-6)
        feat['uniformity'] = np.sum(hist ** 2)
        
        # Smoothness
        variance = np.var(gray)
        feat['smoothness'] = 1 - 1 / (1 + variance)
        
        # === 8. CORNER DETECTION ===
        
        corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        feat['corner_density'] = np.sum(corners > 0.01 * corners.max()) / corners.size
        
        features_list.append(feat)
    
    return features_list

def get_sharpness(mirror_img):
    gray = cv2.cvtColor(mirror_img, cv2.COLOR_BGR2GRAY)


def degradation_features(gray):
    """Cechy do śledzenia degradacji lustra w czasie"""
    # Normalizacja - niezależność od jasności
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 1. Entropia - wzrost = więcej "chaosu"/defektów
    hist = np.histogram(gray_norm, bins=256, range=(0, 256), density=True)[0]
    texture_entropy = entropy(hist + 1e-10)

    # 2. Statystyki gradientów - wykrywają zadrapania/plamy
    grad = sobel(gray_norm.astype(float))
    grad_mean = np.mean(grad)
    grad_std = np.std(grad)

    # 3. GLCM - homogeniczność i korelacja
    glcm = graycomatrix(gray_norm, [1, 3], [0, np.pi / 4, np.pi / 2],
                        256, symmetric=True, normed=True)
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    contrast = graycoprops(glcm, 'contrast').mean()

    # 4. LBP uniformity - spadek = więcej nieregularnych wzorców
    lbp = local_binary_pattern(gray_norm, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, density=True)
    lbp_uniformity = np.max(lbp_hist)  # dominacja jednego wzorca = jednorodność

    return {
        'entropy': texture_entropy,  # ↑ = degradacja
        'grad_mean': grad_mean,  # ↑ = więcej krawędzi/defektów
        'grad_std': grad_std,  # ↑ = więcej defektów
        'homogeneity': homogeneity,  # ↓ = degradacja
        'correlation': correlation,  # ↓ = degradacja
        'contrast': contrast,  # ↑ = degradacja
        'lbp_uniformity': lbp_uniformity  # ↓ = degradacja
    }

def build_hex_mirror_graph(row_sizes):
    """
    Buduje graf sąsiedztwa dla luster w układzie heksagonalnym.
    row_sizes: lista z liczbą luster w kolejnych rzędach,
               np. [14, 16, 18, 20, 22, 24, 26, 28, 26, 24, 22, 20, 18, 16, 14]
    """
    mirrors = {}  # (row, col) -> mirror_id
    positions = {}  # mirror_id -> (x, y) fizyczna pozycja
    mirror_id = 0

    max_width = max(row_sizes)

    for row_idx, n_mirrors in enumerate(row_sizes):
        # Offset - centrowanie każdego rzędu
        offset = (max_width - n_mirrors) / 2.0
        for col_idx in range(n_mirrors):
            mirrors[(row_idx, col_idx)] = mirror_id

            # Pozycja fizyczna (uwzględniając hex staggering)
            x = offset + col_idx
            y = row_idx * np.sqrt(3) / 2  # odstęp hex między rzędami
            positions[mirror_id] = (x, y)
            mirror_id += 1

    total_mirrors = mirror_id
    adjacency = defaultdict(list)

    for row_idx, n_mirrors in enumerate(row_sizes):
        for col_idx in range(n_mirrors):
            mid = mirrors[(row_idx, col_idx)]

            # Sąsiedzi w tym samym rzędzie
            if col_idx > 0:
                adjacency[mid].append(mirrors[(row_idx, col_idx - 1)])
            if col_idx < n_mirrors - 1:
                adjacency[mid].append(mirrors[(row_idx, col_idx + 1)])

            # Sąsiedzi w rzędzie powyżej i poniżej
            for d_row, neighbor_row_idx in [(-1, row_idx - 1), (1, row_idx + 1)]:
                if 0 <= neighbor_row_idx < len(row_sizes):
                    n_neighbor = row_sizes[neighbor_row_idx]
                    delta = (n_neighbor - n_mirrors)

                    # W siatce hex, sąsiedztwo zależy od tego,
                    # czy sąsiedni rząd jest szerszy czy węższy
                    if delta > 0:  # sąsiedni rząd szerszy
                        neighbor_cols = [col_idx, col_idx + 1]
                    elif delta < 0:  # sąsiedni rząd węższy
                        neighbor_cols = [col_idx - 1, col_idx]
                    else:  # ten sam rozmiar
                        neighbor_cols = [col_idx - 1, col_idx, col_idx + 1]

                    for nc in neighbor_cols:
                        if 0 <= nc < n_neighbor:
                            nid = mirrors[(neighbor_row_idx, nc)]
                            if nid not in adjacency[mid]:
                                adjacency[mid].append(nid)

    return adjacency, positions, total_mirrors

def make_square_patch(center, size):
    """Kwadratowe lustro wycentrowane na (center)."""
    return patches.Rectangle(
        (center[0] - size / 2, center[1] - size / 2), size, size
    )
def visualization_mirrors(feature_array, feature_name, ax):
    row_dict = {
        "row 1": 9,
        "row 2": 11,
        "row 3": 13,
        "row 4": 15,
        "row 5": 17,
        "row 6": 17,
        "row 7": 17,
        "row 8": 17,
        "row 9": 17,
        "row 10": 17,
        "row 11": 17,
        "row 12": 17,
        "row 13": 17,
        "row 14": 15,
        "row 15": 13,
        "row 16": 11,
        "row 17": 9,
    }

    list(row_dict.values())
    # Definicja rzędów (symetryczny sześciokąt)
    row_sizes = list(row_dict.values())
    adj, pos, n_total = build_hex_mirror_graph(row_sizes)
    sq_size = 0.85

    print(f"Łącznie luster: {n_total}")
    print(f"Lustro 50 -> sąsiedzi: {adj[1]}")

    # ──────────────────────────────────────────────
    # Wizualizacja 1: Graf sąsiedztwa + kwadraty
    # ──────────────────────────────────────────────

    edge_lines = []
    drawn_edges = set()
    for mid, neighbors in adj.items():
        for nid in neighbors:
            edge_key = tuple(sorted((mid, nid)))
            if edge_key not in drawn_edges:
                drawn_edges.add(edge_key)
                edge_lines.append([pos[mid], pos[nid]])

    lc = LineCollection(edge_lines, colors='#cccccc', linewidths=0.5,
                        zorder=1, alpha=0.6)
    ax.add_collection(lc)

    sq_patches = [make_square_patch(pos[mid], sq_size) for mid in range(n_total)]

    pc = PatchCollection(sq_patches, cmap='YlOrRd', edgecolors='#333333',
                         linewidths=0.8, zorder=2)

    pc.set_array(np.array(feature_array))

    pc.set_clim(min(feature_array), max(feature_array))
    ax.add_collection(pc)

    for mid in range(n_total):
        x, y = pos[mid]
        ax.text(x, y, str(mid), ha='center', va='center',
                fontsize=6.5, fontweight='bold', color='#222222', zorder=3)

    cbar = plt.colorbar(pc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(feature_name, fontsize=12)

    ax.set_xlim(-1, max(row_sizes) + 1)
    ax.set_ylim(16, -1)
    ax.set_aspect('equal')
    ax.set_title("MAGIC Mirrors")
    ax.set_xlabel('Column', fontsize=11)
    ax.set_ylabel('Row', fontsize=11)
    ax.grid(False)
    ax.set_facecolor('#f8f8f8')
    plt.tight_layout()

    plt.show()