import h5py
import yaml

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_features_array(path_to_h5_baseline_file):
    with h5py.File(path_to_h5_baseline_file, "r") as f:
        features = f["feature_matrix"][:]        # (n_images, 249, 11)
    return features