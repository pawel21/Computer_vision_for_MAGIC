from PIL import Image
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        f_a = get_texture_feat(img_a)
        f_b = get_texture_feat(img_b)

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


def find_outlier_mirrors(feature_diffs, n_top=5):
    """Znajdź lustra najbardziej odstające."""
    # Sortuj po odległości malejąco
    sorted_mirrors = sorted(feature_diffs, key=lambda x: x['distance'], reverse=True)

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