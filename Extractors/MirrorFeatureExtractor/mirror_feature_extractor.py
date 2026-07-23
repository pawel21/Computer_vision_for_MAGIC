import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.path import Path
from PIL import Image
import cv2
import pickle 
import glob
import os

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    label,
    laplace,
    sobel,
)
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.decomposition import PCA



from MirrorExtractor.simple_mirror_extractor import SimpleMirrorExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

def extract_photometric_features(gray_img, mask):
    """Extract illumination and contrast statistics over the masked facet.

    Contrast metrics are normalised by the median, making them dimensionless
    and insensitive to the overall illumination level.

    Features:
        - median: Median brightness. Proxy for scene illumination.
        - cov: Coefficient of variation (std/mean). Primary AMC lock-loss
          discriminator: a defocused facet reflects near-uniform sky, so this
          collapses towards zero.
        - robust_cov: Outlier-resistant cov (scaled MAD / median). Preferred
          on small patches where a single hot pixel skews std.
        - iqr_norm, rng_norm: Normalised interquartile and 5-95 percentile
          spread. Alternative contrast measures with different outlier
          sensitivity.
        - sat_frac: Fraction of pixels at or above 250 DN. This is a QUALITY
          FLAG, not a feature: sensor saturation also drives cov towards zero
          and is the dominant false-positive mode for lock-loss detection.
          Log it separately and exclude affected frames downstream.

    Args:
        gray_img: 2D array, cropped grayscale patch.
        mask: 2D bool array, True inside the facet.

    Returns:
        dict: Feature name -> float. All np.nan on failure.
    """
    names = ['median', 'cov', 'robust_cov', 'iqr_norm', 'rng_norm', 'sat_frac']
    features = {n: np.nan for n in names}

    try:
        v = gray_img[mask].astype(np.float64)
        if v.size < 10:
            logging.warning("Photometric: fewer than 10 masked pixels")
            return features

        med = np.median(v)
        mad = np.median(np.abs(v - med))
        p05, p25, p75, p95 = np.percentile(v, [5, 25, 75, 95])
        eps = 1e-8

        features['median'] = float(med)
        features['cov'] = float(v.std() / (v.mean() + eps))
        features['robust_cov'] = float(1.4826 * mad / (med + eps))
        features['iqr_norm'] = float((p75 - p25) / (med + eps))
        features['rng_norm'] = float((p95 - p05) / (med + eps))
        features['sat_frac'] = float((v >= 250).mean())

    except Exception as e:
        logging.warning(f"Photometric feature extraction failed: {e}")

    return features

def new_extract_edge_features(gray_img, mask):
    """Extract gradient and edge features over the masked facet.

    Measures how much structural detail the facet reflects. A defocused facet
    averages over a wide solid angle, washing out scene edges.

    Neighbourhood operators read a 3x3 window, so two corrections are applied:
    the background is replaced by nearest-neighbour extrapolation from inside
    the mask (removing the artificial 0 -> 200 DN step at the border), and
    statistics are averaged over the eroded mask only.

    Note that sobel_mean and laplacian_* scale with overall brightness. If they
    prove unstable across frames, divide them by the median brightness.

    Features:
        - sobel_mean: Mean first-order gradient magnitude.
        - laplacian_mean, laplacian_std: Mean and spread of the absolute
          second derivative. Sensitive to blur.
        - edge_density: Fraction of eroded-mask pixels flagged by Canny.
          Hard-coded thresholds may yield very few pixels on small patches;
          check its frame-to-frame stability before relying on it.

    Args:
        gray_img: 2D array, cropped grayscale patch.
        mask: 2D bool array, True inside the facet.

    Returns:
        dict: Feature name -> float. All np.nan on failure.
    """
    names = ['sobel_mean', 'laplacian_mean', 'laplacian_std', 'edge_density']
    features = {n: np.nan for n in names}

    try:
        core = binary_erosion(mask, np.ones((3, 3), bool))
        if core.sum() < 10:
            logging.warning("Edge: mask core too small after erosion")
            return features

        # Replace background with the nearest in-mask value: no artificial step.
        idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        g = gray_img[tuple(idx)].astype(np.float64)

        sx, sy = sobel(g, axis=1), sobel(g, axis=0)
        features['sobel_mean'] = float(np.sqrt(sx ** 2 + sy ** 2)[core].mean())

        lap = np.abs(laplace(g))
        features['laplacian_mean'] = float(lap[core].mean())
        features['laplacian_std'] = float(lap[core].std())

        edges = cv2.Canny(np.clip(g, 0, 255).astype(np.uint8), 50, 150)
        features['edge_density'] = float((edges[core] > 0).sum() / core.sum())

    except Exception as e:
        logging.warning(f"Edge feature extraction failed: {e}")

    return features

def new_extract_glcm_features(gray_img, mask, levels=16):
    """Extract texture features from the Gray-Level Co-occurrence Matrix.

    Computed at distance 1 over four angles (0, 45, 90, 135 degrees) and
    averaged for rotation invariance.

    Masking works by reserving quantisation level 0 for background and slicing
    it away afterwards; graycoprops renormalises the matrix, so only in-facet
    pixel pairs contribute.

    Quantisation uses a FIXED 0-255 range, deliberately not per-patch
    stretching. Stretching would rescale sensor noise on a near-uniform
    defocused patch into apparent texture, destroying the target signal.

    Features:
        - glcm_contrast: Squared intensity difference between neighbours.
        - glcm_dissimilarity: As above, linearly weighted.
        - glcm_homogeneity: Concentration near the GLCM diagonal.
        - glcm_energy: Sum of squared entries. Biased by roughly 1/N on sparse
          matrices; the bias is constant for a fixed mask and absorbed by a
          per-facet temporal baseline, but it inflates frame-to-frame variance.
        - glcm_correlation: Linear predictability of neighbouring gray levels.
          The least redundant of the five, as it measures structure
          predictability rather than magnitude.

    Contrast and dissimilarity largely duplicate sobel_mean. Check the
    correlation matrix and keep one.

    Args:
        gray_img: 2D array, cropped grayscale patch.
        mask: 2D bool array, True inside the facet.
        levels: Quantisation levels. With ~230 masked pixels, 8 gives denser
            bins and 16 finer resolution; pick empirically by stability.

    Returns:
        dict: Feature name -> float. All np.nan on failure.
    """
    features = {f"glcm_{p}": np.nan for p in GLCM_PROPERTIES}

    try:
        q = np.clip(gray_img.astype(np.float64) / 256.0 * levels, 0, levels - 1)
        q = q.astype(np.uint8) + 1          # shift to 1..levels
        q[~mask] = 0                        # level 0 reserved for background

        glcm = graycomatrix(
            q,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=levels + 1,
            symmetric=True,
        )
        glcm = glcm[1:, 1:, :, :]           # discard the background row/column

        if glcm.sum() == 0:
            logging.warning("GLCM: no valid pixel pairs after masking")
            return features

        for prop in GLCM_PROPERTIES:
            features[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())

    except Exception as e:
        logging.warning(f"GLCM extraction failed: {e}")

    return features

def new_extract_lbp_features(gray_img, mask, points=8, radius=1):
    """Extract texture features from uniform Local Binary Patterns.

    LBP thresholds each pixel against its circular neighbours and keeps only
    the SIGNS of the differences. This makes it robust to illumination scaling
    but blind to contrast amplitude.

    Consequence for defocus: a near-uniform patch still yields sign patterns
    driven by sensor noise. Random signs favour non-uniform patterns (only 58
    of 256 codes are uniform), so at lock-loss lbp_nonuniform RISES towards
    ~0.77 and lbp_entropy FALLS. This is the opposite direction from gradual
    soiling, where reflected structure becomes more irregular.

    Expect LBP to underperform cov on lock-loss (it discards the amplitude
    carrying the signal) but to remain more stable under illumination changes,
    making it useful for suppressing weather-driven false positives.

    Features:
        - lbp_entropy: Shannon entropy of the pattern histogram, in bits.
          Roughly 1.5 for pure noise, 2.5+ for structured reflections.
        - lbp_nonuniform: Share of non-uniform patterns (bin P+1). Preferred
          over max-bin uniformity, which silently switches which bin it tracks
          and can jump without any change in the data.

    Args:
        gray_img: 2D array, cropped grayscale patch.
        mask: 2D bool array, True inside the facet.
        points: Number of circular neighbours.
        radius: Circle radius in pixels.

    Returns:
        dict: Feature name -> float. All np.nan on failure.
    """
    features = {'lbp_entropy': np.nan, 'lbp_nonuniform': np.nan}
    n_bins = points + 2                     # skimage 'uniform' returns 0..P+1

    try:
        k = 2 * radius + 1
        core = binary_erosion(mask, np.ones((k, k), bool))
        if core.sum() < 20:
            logging.warning("LBP: mask core too small after erosion")
            return features

        lbp = local_binary_pattern(gray_img, P=points, R=radius, method='uniform')
        hist = np.bincount(lbp[core].astype(int), minlength=n_bins).astype(np.float64)
        hist /= hist.sum()

        nz = hist > 0
        features['lbp_entropy'] = float(-np.sum(hist[nz] * np.log2(hist[nz])))
        features['lbp_nonuniform'] = float(hist[points + 1])

    except Exception as e:
        logging.warning(f"LBP extraction failed: {e}")

    return features

def build_mask(gray_img, thr=0):
    """Derive the facet mask from a cropped patch.

    The crop is a bounding box around a hexagonal facet, so its corners are
    background. This returns a boolean mask marking the mirror surface.

    Compute this ONCE per facet and reuse it for every frame. Recomputing it
    per frame makes the pixel count drift with scene content, which mimics a
    slow change in the mirror itself.

    Args:
        gray_img: 2D array, cropped grayscale patch.
        thr: Background threshold. Raise to ~10 if JPEG ringing leaves
            non-zero values along the crop corners.

    Returns:
        2D bool array, True inside the facet.
    """
    m = binary_fill_holes(gray_img > thr)   # keep genuinely dark reflected pixels
    lab, n = label(m)
    if n > 1:                               # drop isolated specks, keep the facet
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == sizes.argmax()
    return m

def extract_glcm_features(gray_img):
    """Extract texture features using Gray-Level Co-occurrence Matrix (GLCM).

    Computes GLCM at distance=1 across four angles (0°, 45°, 90°, 135°)
    and averages the results to produce rotation-invariant features.

    Features extracted:
        - glcm_contrast: intensity difference between neighboring pixels.
          Higher = rougher texture, potential surface degradation.
        - glcm_dissimilarity: similar to contrast but linear weighting.
          Higher = more local variation in reflection.
        - glcm_homogeneity: closeness of GLCM values to diagonal.
          Higher = more uniform reflection (smooth mirror).
        - glcm_energy: sum of squared GLCM entries.
          Higher = more repetitive/ordered texture pattern.
        - glcm_correlation: linear dependency of gray levels on neighbors.
          Higher = more predictable spatial structure.

    Args:
        gray_img: 2D numpy array, grayscale mirror patch (typically ~80x90 px).
                  Will be cast to uint8 internally for GLCM computation.

    Returns:
        dict: Feature name -> float value. On failure, all values are np.nan.
    """
    features = {}

    try:
        angles_list = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(
            gray_img.astype(np.uint8),
            distances=[1],
            angles=angles_list,
            levels=256)
        for prop in GLCM_PROPERTIES:
            features[f"glcm_{prop}"] = graycoprops(glcm, prop).mean()

    except Exception as e:
        logging.warning(f"GLCM extraction failed with exception: {e}")
        for prop in GLCM_PROPERTIES:
            features[f"glcm_{prop}"] = np.nan

    return features


def extract_lbp_features(gray_img, points=8, radius=1):
    """Extract texture features using Local Binary Pattern (LBP).

    Computes uniform LBP and derives entropy and uniformity metrics
    from the pattern histogram. Uniform LBP captures local micro-textures
    (edges, corners, flat regions) — for mirror patches, higher entropy
    indicates more complex/irregular reflection texture (potential degradation),
    while higher uniformity suggests smooth, consistent reflection.

    Features extracted:
        - lbp_entropy: Shannon entropy of the LBP histogram.
          Higher = more diverse local patterns, less uniform surface.
        - lbp_uniformity: max bin proportion in the histogram.
          Higher = one dominant pattern (typically flat regions on good mirrors).

    Args:
        gray_img: 2D numpy array, grayscale mirror patch (typically ~80x90 px).
        points: number of neighbors in LBP circle (default 8).
        radius: radius of LBP circle in pixels (default 1).

    Returns:
        dict: Feature name -> float value. On failure, all values are np.nan.
    """
    n_bins = points * (points - 1) + 3  # uniform LBP: P*(P-1)+3 unique patterns
    features = {}

    try:
        lbp = local_binary_pattern(gray_img, P=points, R=radius, method='uniform')

        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

        # Shannon entropy — miara złożoności tekstury
        features['lbp_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))

        # Uniformity — jak bardzo dominuje jeden wzorzec
        features['lbp_uniformity'] = hist.max()

    except Exception as e:
        logging.warning(f"LBP extraction failed: {e}")
        features['lbp_entropy'] = np.nan
        features['lbp_uniformity'] = np.nan

    return features


def extract_edge_features(gray_img):
    """Extract edge and gradient features using Sobel, Laplacian, and Canny.

    Measures sharpness and structure of reflected image on mirror patches.
    Sharp, clear reflections indicate good mirror condition — degraded or
    dirty mirrors produce blurred, low-contrast reflections with weaker
    edge responses.

    Features extracted:
        - sobel_mean: mean gradient magnitude (Sobel).
          Higher = sharper reflection with more defined edges.
        - laplacian_mean: mean absolute second derivative.
          Higher = more rapid intensity changes, sharper details.
        - laplacian_std: std of absolute second derivative.
          Higher = more variation in sharpness across the patch.
        - edge_density: fraction of pixels detected as edges (Canny).
          Higher = more structural detail visible in reflection.

    Args:
        gray_img: 2D numpy array, grayscale mirror patch (typically ~80x90 px).
                  Will be cast to uint8 internally for Canny detection.

    Returns:
        dict: Feature name -> float value. On failure, all values are np.nan.
    """
    feature_names = ['sobel_mean', 'laplacian_mean', 'laplacian_std', 'edge_density']
    features = {}

    try:
        # Sobel — gradient pierwszego rzędu
        sobelx = sobel(gray_img, axis=1)
        sobely = sobel(gray_img, axis=0)
        sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        features['sobel_mean'] = np.mean(sobel_magnitude)

        # Laplacian — gradient drugiego rzędu, wrażliwy na rozmycie
        lap = np.abs(laplace(gray_img))
        features['laplacian_mean'] = np.mean(lap)
        features['laplacian_std'] = np.std(lap)

        # Canny — binarna mapa krawędzi, gęstość = ile struktury widać
        edges = cv2.Canny(gray_img.astype(np.uint8), 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size

    except Exception as e:
        logging.warning(f"Edge feature extraction failed: {e}")
        for name in feature_names:
            features[name] = np.nan

    return features

def extract_brightness_features(gray_img):
    features = {}
    mean = np.mean(gray_img)
    std = np.std(gray_img)
    features['brightness_mean'] = mean
    features['brightness_cov'] = std / (mean + 1e-8)
    return features