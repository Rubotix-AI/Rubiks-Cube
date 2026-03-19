import cv2
import numpy as np

def extract_face(frame):
    # create circle centre positions
    centres = [(40,40), (110,110), (180, 180), (40, 110), (110, 40), (40, 180), (180, 40), (110, 180), (180, 110)]
    
    cube_face = []
    for i in centres:
        col = extract_color_from_circle(frame, centres[i])
        cube_face.append(col)

    return cube_face

def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)

def extract_color_from_circle(frame, centre):
    mask = np.zeros(frame.shape[:2], dtype='uint8')
    cv2.circle(mask, centre, 25, 255, -1)
    mask = cv2.bitwise_and(frame, frame, mask=mask)
    max_col = bincount_app(frame)

    return max_col