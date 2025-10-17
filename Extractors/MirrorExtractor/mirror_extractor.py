import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.path import Path
from PIL import Image
import cv2
import pickle 
import glob
import os


class MirrorExtractor:

    def __init__(self, path_to_points):
        self.path_to_points = path_to_points
        self.m_points = get_matrix_points(self.path_to_points)
    
    def get_matrix_points(self, crossing_points_path):
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

    def get_coords(self, idx):
        rows, cols = self.m_points.shape
        indices = [(i, j) for i in range(rows - 1) for j in range(cols - 1)]
        N = len(indices)
        state = {'idx': idx}
        i, j = indices[state['idx']]
        x_coords = [
            self.m_points[i, j]['x'],
            self.m_points[i, j+1]['x'],
            self.m_points[i+1, j+1]['x'],
            self.m_points[i+1, j]['x'],
            self.m_points[i, j]['x']  # close the loop
        ]
        y_coords = [
            self.m_points[i, j]['y'],
            self.m_points[i, j+1]['y'],
            self.m_points[i+1, j+1]['y'],
            self.m_points[i+1, j]['y'],
            self.m_points[i, j]['y']
        ]
        return x_coords, y_coords

    @staticmethod
    def extract_polygon_region_cv2(img_array, x_coords, y_coords):
        """
        Extracts a polygonal region from an image and returns it as a new (cropped) image.
        Args:
            img: Input image array.
            x_coords, y_coords: Lists of x and y coordinates of polygon vertices.
    
        Returns:
            Cropped polygon region as a new image.
        """
        img = img_array
        
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

    def get_all_mirrors(img_array):
        mirror_list = []
        for i in range(289):
            x_coords, y_coords = extractor.get_coords(288)
            cropped = extractor.extract_polygon_region_cv2(img_array, x_coords, y_coords)
            mirror_list.append(cropped)
        return mirror_list
        
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

def get_matrix_points(crossing_points_path='data/crossings_points.pkl'):
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

def show_mirror(path, idx):
    m_points = get_matrix_points()            
    rows, cols = m_points.shape
    indices = [(i, j) for i in range(rows - 1) for j in range(cols - 1)]
    N = len(indices)
    state = {'idx': idx}
    
    i, j = indices[state['idx']]

    x_coords, y_coords = get_coords(state, indices, m_points)

    img_np = np.array(Image.open(path).convert('RGB'))
    cropped = extract_polygon_region_cv2(path, x_coords, y_coords)

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.imshow(img_np)
    ax1.plot(x_coords, y_coords, 'r-', lw=2)
    ax1.scatter(x_coords[:-1], y_coords[:-1], c='cyan', s=30)
    ax1.set_title(f"Mirror ({i}, {j})")
    ax1.axis('off')

    ax2.imshow(cropped)
    ax2.set_title("Cropped Region")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def add_box_around_mirror(img_np, list_idx):
    m_points = get_matrix_points()            
    rows, cols = m_points.shape
    indices = [(i, j) for i in range(rows - 1) for j in range(cols - 1)]
    N = len(indices)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10,5))
    ax1.imshow(img_np)
    for idx in list_idx:
        state = {'idx': idx}
        i, j = indices[state['idx']]
        x_coords, y_coords = get_coords(state, indices, m_points)
        ax1.plot(x_coords, y_coords, 'r-', lw=0.5)
        ax1.scatter(x_coords[:-1], y_coords[:-1], c='cyan', s=10)
    
    ax1.axis('off')


    plt.tight_layout()
    plt.show()


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
        mirror = extract_polygon_reg