import cv2
from skimage import feature
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA
import numpy as np

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