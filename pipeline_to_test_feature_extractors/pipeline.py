import numpy as np
import h5py
from feature_extractors import raw_pixels, hog_features
from clustering import cluster_dbscan



def process_images(h5file, extractor_fn, cluster_fn, mirror_id):
    feats = []

    with h5py.File(h5file, 'r') as f:
        for entry_name in f["images"]:
            img = f["images"][entry_name]["mirrors"][mirror_id][:]
            feat = extractor_fn(img)
            feats.append(feat)

        features = np.stack(feats)
        labels = cluster_fn(features)
        return labels, features