import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.path import Path
from PIL import Image
import cv2
import pickle 
import glob
import os

def extract_polygon_region_cv2(img_path, x_coords, y_coords):
    """
    Extracts a polygonal region from an image and returns it as a new (cropped) image.
    Args:
        img: Input path to image.
        x_coords, y_coords: Lists of x and y coordinates of polygon vertices.

    Returns:
        Cropped polygon region as a new image.
    """
    img = np.array(Image.open(img_path).convert('RGB'))
    
    # Convert polygon points to int32 and to shape (N, 1, 2)
    pts = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Create a mask with the same shape as the image (1 channel)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=255)

    # Apply mask to image
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Find bounding rectangle and crop
    x, y, w, h = cv2.boundingRect(pts)
    cropped = masked_img[y:y+h, x:x+w]

    return cropped

def create_dir_with_mirrors(image_list, idx):
    with open('../data/crossings_points.pkl', 'rb') as f:
        crossing_points = pickle.load(f)

    dt = np.dtype([('x', 'i4'), ('y', 'i4')])
    m_points = np.zeros((18, 18), dtype=dt)
    i = 0 
    j = 0
    for p in crossing_points:
        m_points[i, j] = p
        j += 1
        if j%18 == 0:
            j = 0
            i += 1
            
    rows, cols = m_points.shape
    indices = [(i, j) for i in range(rows - 1) for j in range(cols - 1)]
    N = len(indices)
    state = {'idx': idx}

    x, y = get_coords(state, indices, m_points)

    PATH_TO_MIRROR_DIR = f"/media/pgliwny/ADATA HD330/Computer_Vision_system/data/MAGIC/webcam/mirrors/mirror_test"
    
    for path_img in image_list:    
        mirror = extract_polygon_region_cv2(path_img, x, y)
        new_name = f"m_{idx}" + "_" + path_img.split("/")[-1] 
        new_path = os.path.join(PATH_TO_MIRROR_DIR, new_name)
        cv2.imwrite(new_path, mirror)

def get_coords(state, indices, m_points):
    i, j = indices[state['idx']]
    x_coords = [
        m_points[i, j]['x'],
        m_points[i, j+1]['x'],
        m_points[i+1, j+1]['x'],
        m_points[i+1, j]['x'],
        m_points[i, j]['x']  # close the loop
    ]
    y_coords = [
        m_points[i, j]['y'],
        m_points[i, j+1]['y'],
        m_points[i+1, j+1]['y'],
        m_points[i+1, j]['y'],
        m_points[i, j]['y']
    ]
    return x_coords, y_coords

def get_matrix_points(crossing_points_path='../data/crossings_points.pkl'):
    with open(crossing_points_path, 'rb') as f:
        crossing_points = pickle.load(f)

    dt = np.dtype([('x', 'i4'), ('y', 'i4')])
    m_points = np.zeros((18, 18), dtype=dt)
    i = 0 
    j = 0
    for p in crossing_points:
        m_points[i, j] = p
        j += 1
        if j%18 == 0:
            j = 0
            i += 1
    return m_points