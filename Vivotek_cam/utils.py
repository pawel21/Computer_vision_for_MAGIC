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
    with open('data/crossings_points_IRCamM1T20250702_161000M.pkl', 'rb') as f:
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

def get_matrix_points(crossing_points_path='data/crossings_points_IRCamM1T20250702_161000M.pkl'):
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



def get_mirror_from_image_color(m_points, idx, image):
    ROWS, COLS = 18, 18
    indices = [(i, j) for i in range(ROWS - 1) for j in range(COLS - 1)]
    N = len(indices)
    state = {'idx': idx}

    x, y = get_coords(state, indices, m_points)

    mirror = extract_polygon_region_cv2_from_image_color(image, x, y)
    return mirror
    
def extract_polygon_region_cv2_from_image_color(img, x_coords, y_coords):
    """
    Extracts a polygonal region from a COLOR image and returns it as a new (cropped) image.
    
    Args:
        img: Input color image (NumPy array).
        x_coords, y_coords: Lists of x and y coordinates of polygon vertices.

    Returns:
        Cropped polygon region as a new image.
    """
    # Make sure the input is a NumPy array
    img = np.array(img)
    
    # Create polygon vertex points
    pts = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # --- THIS IS THE ONLY LINE THAT CHANGED ---
    # Create a 2D mask (H, W), not a 3D one, by using img.shape[:2]
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Draw the filled polygon on the mask
    cv2.fillPoly(mask, [pts], color=255)
    
    # Apply the mask to the original color image
    # This zeroes out everything outside the polygon
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    # Crop the image to the bounding rectangle of the polygon
    x, y, w, h = cv2.boundingRect(pts)
    cropped = masked_img[y:y+h, x:x+w]

    return cropped
    
def extract_color_features(image):
    """
    Konwertuje obraz do przestrzeni HSV i oblicza histogram kolorów.
    
    Args:
        image (numpy.ndarray): Obraz w formacie BGR (jak wczytuje OpenCV).
        
    Returns:
        numpy.ndarray: Spłaszczony wektor cech z histogramu HSV.
    """
    # Konwersja obrazu z BGR do HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Obliczenie histogramu dla kanałów H, S, V
    # Definiujemy liczbę przedziałów (bins) dla każdego kanału
    # H: 0-179 (barwa), S: 0-255 (nasycenie), V: 0-255 (jasność)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    
    # Normalizacja histogramu, aby wartości były w zakresie od 0 do 1
    cv2.normalize(hist, hist)
    
    # Spłaszczenie histogramu do jednowymiarowego wektora cech
    return hist.flatten()

def extract_texture_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist