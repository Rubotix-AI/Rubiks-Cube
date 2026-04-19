import cv2
from initialize.utils import bgr2lab, CIEDE2000, getDominantColor
from initialize.config import BGR_COLOURS, THRESH_COL_DELTA, SQUARE_AREA_MIN

def initialProcess(frame):
    converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    noiseless_frame = cv2.fastNlMeansDenoising(converted_frame, None, 20, 7, 7)
    blurred_frame = cv2.blur(noiseless_frame, (3,3))
    canny_frame = cv2.Canny(blurred_frame, 30, 60, 3)
    dilated_frame = cv2.dilate(canny_frame, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

    return dilated_frame

def findFace(frame):
    cp = frame.copy()

    frame = initialProcess(frame=frame)
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    possible_squares = []

    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)

        # if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        #     ratio = w / h
        area = cv2.contourArea(approx)
        #     if SQUARE_RATIO_BOUNDS[0] <= ratio <= SQUARE_RATIO_BOUNDS[1] and area >= SQUARE_AREA_MIN:
        if area >= SQUARE_AREA_MIN:
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
    
    return possible_squares

def windowProcessing(possible_squares):
    """
    update conf levels for each cubelet in cube struct
    
    """