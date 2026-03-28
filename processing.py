import cv2
import numpy as np
from sklearn.cluster import KMeans

from utils import bgr2lab, CIEDE2000
from config import BGR_COLOURS, THRESH_COL_DELTA, SQUARE_RATIO_BOUNDS, SQUARE_AREA_MIN, RANDOM_STATE, KMEANS_K, CELL_CORRECTION_FACTOR

def getDominantColor(frame):
    frame = np.array(frame)
    frame = np.float32(frame.reshape((-1, 3)))

    k = KMEANS_K
    random_state = RANDOM_STATE
    model = KMeans(n_clusters=k, random_state=random_state)
    model.fit(frame)

    dominant_color = model.cluster_centers_.astype(int)
    counts = np.unique(model.labels_, return_counts=True)[1]
    props = counts / counts.sum()

    sorted = np.argsort(-props) # minus cause decreasing order
    most_dom = dominant_color[sorted[0]]

    return most_dom

def initialProcess(frame):
    converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    noiseless_frame = cv2.fastNlMeansDenoising(converted_frame, None, 20, 7, 7)
    blurred_frame = cv2.blur(noiseless_frame, (3,3))
    canny_frame = cv2.Canny(blurred_frame, 30, 60, 3)
    dilated_frame = cv2.dilate(canny_frame, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

    return dilated_frame

def findFace(frame, id):
    cp = frame.copy()

    frame = initialProcess(frame=frame)
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    possible_squares = []
    face = []

    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / h
            area = cv2.contourArea(approx)

            if SQUARE_RATIO_BOUNDS[0] <= ratio <= SQUARE_RATIO_BOUNDS[1] and area >= SQUARE_AREA_MIN:
                most_dom = getDominantColor(frame=cp[y:y+h, x:x+w])
                lab_dom = bgr2lab(most_dom)

                color_deltas = []
                for rubik_col, rubik_rgb in BGR_COLOURS.items():
                    rubik_lab = bgr2lab(rubik_rgb)
                    diff = CIEDE2000(lab_dom, rubik_lab)

                    color_deltas.append({"colour": rubik_col, "delta": diff})

                
                best_col_delta = min(color_deltas, key=lambda item: item["delta"])

                if best_col_delta["delta"] < THRESH_COL_DELTA:
                    possible_squares.append({"x": x, "y": y, "w": w, "h": h, "colour": best_col_delta["colour"]})
    
    if len(possible_squares) == 9:
        squares_sorted_x = sorted(possible_squares, key=lambda item: item["x"])
        squares_sorted_y = sorted(possible_squares, key=lambda item: item["y"])

        for i in [0, 3, 6]:
            unsorted_y = [squares_sorted_y[i], squares_sorted_y[i+1], squares_sorted_y[i+2]]
            sorted_y = sorted(unsorted_y, key=lambda item: item["x"])

            face.extend(sorted_y)

        centre_cell = face[4]
        x_min = squares_sorted_x[0]
        x_max = squares_sorted_x[-1]
        y_min = squares_sorted_y[0]
        y_max = squares_sorted_y[-1]

        gap_width = int(centre_cell["w"] * CELL_CORRECTION_FACTOR)
        gap_height = int(centre_cell["h"] * CELL_CORRECTION_FACTOR)

        if (centre_cell["x"] - x_min["x"] <= gap_width) and (x_max["x"] - centre_cell["x"] <= gap_width):
            if (centre_cell["y"] - y_min["y"] <= gap_height) and (y_max["y"] - centre_cell["y"] <= gap_height):
                pos = id
                print(face)
                for square in face:
                    x, y, w, h = square["x"], square["y"], square["w"], square["h"]
                    cv2.rectangle(cp, (x, y), (x+w, y+h), (220, 220, 220), 5)
                    cv2.putText(cp, f"{pos}", (x+int(0.5*w), y+int(0.5*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                    pos += 1

    return face, cp