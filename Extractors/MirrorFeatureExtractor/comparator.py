from PIL import Image
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from MirrorFeatureExtractor.mirror_feature_extractor import MirrorFeatureExtractor


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
    cropped = masked_img[y: y +h, x: x +w]
    cropped_gray = rgb_to_gray(cropped)
    return cropped_gray

RGB_TO_GRAY = np.array([0.299, 0.587, 0.114])

def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """Konwersja RGB do skali szarośći metodą weighted average"""
    return np.dot(img_rgb[..., :3], RGB_TO_GRAY).astype(np.uint8)


def get_diff_all_features(mirrors_list_1, mirrors_list_2):
    feature_diffs = []

    for mirror_idx, (img_a, img_b) in enumerate(zip(mirrors_list_1, mirrors_list_2)):
        #f_a = get_texture_feat(img_a)
        #f_b = get_texture_feat(img_b)
        f_a = extract_texture_features(img_a)
        f_b = extract_texture_features(img_b)
        # Zamień słowniki na wektory
        vec_a = np.array(list(f_a.values()))
        vec_b = np.array(list(f_b.values()))

        feature_diffs.append({
            'mirror_idx': mirror_idx,
            'vec_a': vec_a,
            'vec_b': vec_b,
            'feature_names': list(f_a.keys())
        })

    # Normalizacja - ważne bo cechy mają różne skale!
    all_vecs = np.vstack([fd['vec_a'] for fd in feature_diffs] +
                         [fd['vec_b'] for fd in feature_diffs])
    scaler = StandardScaler()
    scaler.fit(all_vecs)

    # Oblicz znormalizowane odległości
    for fd in feature_diffs:
        vec_a_norm = scaler.transform([fd['vec_a']])[0]
        vec_b_norm = scaler.transform([fd['vec_b']])[0]

        # Odległość euklidesowa
        distance = np.linalg.norm(vec_a_norm - vec_b_norm)
        fd['distance'] = distance
        fd['diff_vector'] = vec_a_norm - vec_b_norm

    return feature_diffs

def get_texture_feat(gray_img):
    feat_extractor = MirrorFeatureExtractor()
    gray_norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    feat_dict = feat_extractor.extract_texture_features(gray_norm)
    return feat_dict

def get_edge_and_gradient_feat(gray_img):
    feat_extractor = MirrorFeatureExtractor()
    gray_norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    feat_dict = feat_extractor.extract_edge_and_gradient_features(gray_norm)
    return feat_dict


def find_outlier_mirrors(feature_diffs, n_top=5):
    """Znajdź lustra najbardziej odstające."""
    # Sortuj po odległości malejąco
    sorted_mirrors = sorted(feature_diffs, key=lambda x: x['distance'], reverse=True)
    text = ""
    print("=== Lustra najbardziej odstające ===")
    for i, mirror in enumerate(sorted_mirrors[:n_top]):
        print(f"\n#{ i +1} Lustro {mirror['mirror_idx']}: distance = {mirror['distance']:.4f}")

        # Pokaż które cechy najbardziej się różnią
        diff = np.abs(mirror['diff_vector'])
        top_feat_idx = np.argsort(diff)[::-1][:3]
        print("   Największe różnice w cechach:")
        for idx in top_feat_idx:
            print(f"   - {mirror['feature_names'][idx]}: {diff[idx]:.4f}")

    return sorted_mirrors

def get_outlier_mirrors_report(feature_diffs, n_top=5):
    """Znajdź lustra najbardziej odstające."""
    # Sortuj po odległości malejąco
    sorted_mirrors = sorted(feature_diffs, key=lambda x: x['distance'], reverse=True)

    text = "=== Lustra najbardziej odstające ==="
    for i, mirror in enumerate(sorted_mirrors[:n_top]):
        text += f"\n#{ i +1} Lustro {mirror['mirror_idx']}: distance = {mirror['distance']:.4f}"

        # Pokaż które cechy najbardziej się różnią
        diff = np.abs(mirror['diff_vector'])
        top_feat_idx = np.argsort(diff)[::-1][:3]
        text += "   Największe różnice w cechach:"
        for idx in top_feat_idx:
            text += f"   - {mirror['feature_names'][idx]}: {diff[idx]:.4f}"
    return text


def extract_texture_features(gray_img):
    features = {}

    # LBP - multi-scale
    for R in [1, 2, 3]:
        P = 8 * R
        lbp = local_binary_pattern(gray_img, P=P, R=R, method='uniform')
        n_bins = P + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        features[f'lbp_R{R}_mean'] = np.mean(lbp)
        features[f'lbp_R{R}_std'] = np.std(lbp)
        features[f'lbp_R{R}_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))

    # GLCM - multiple angles, averaged
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    distances = [1, 3]
    glcm = graycomatrix(gray_img.astype(np.uint8),
                        distances=distances, angles=angles, levels=256)
    for prop in ['contrast', 'dissimilarity', 'homogeneity',
                 'energy', 'correlation']:
        vals = graycoprops(glcm, prop)
        features[f'glcm_{prop}_mean'] = np.mean(vals)
        features[f'glcm_{prop}_std'] = np.std(vals)

    # Edge features
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    features['edge_mean'] = np.mean(gradient_mag)
    features['edge_std'] = np.std(gradient_mag)
    edges = cv2.Canny(gray_img, 50, 150)
    features['edge_density'] = np.sum(edges > 0) / edges.size

    # Basic intensity stats
    features['intensity_mean'] = np.mean(gray_img)
    features['intensity_std'] = np.std(gray_img)

    return features