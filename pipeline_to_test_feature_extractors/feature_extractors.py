import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

def raw_pixels(image):
    return image.flatten() / 255.0

def hog_features(image):
    return hog(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)

def lbp_histogram(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
    return hist

def pca_features(images, n_components=10):
    flat = [img.flatten() / 255.0 for img in images]
    return PCA(n_components=n_components).fit_transform(flat)

mobilenet = MobileNetV2(include_top=False, input_shape=(64, 64, 3), pooling='avg')

def cnn_features(image):
    resized = cv2.resize(image, (64, 64))
    rgb = np.repeat(resized[:, :, np.newaxis], 3, axis=2)
    x = preprocess_input(img_to_array(rgb))
    return mobilenet.predict(x[np.newaxis])[0]
