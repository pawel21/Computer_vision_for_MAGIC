import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.path import Path
from PIL import Image
import cv2
import pickle 
import glob
import os


class MirrorFeatureExtractor:

    def __init__(self):
        pass

    @staticmethod
    def extract_statistical_features(self):
        """
        Extract statistical features from a single mirror image

        Args:
            mirror_img: numpy array of shape (H, W, 3)

        Returns:
            dict: Dictionary of feature name -> feature value
        """
        features = {}

        # Statistical features
        features['skewness'] = skew(mirror_img.flatten())
        features['kurtosis'] = kurtosis(mirror_img.flatten())

        return features