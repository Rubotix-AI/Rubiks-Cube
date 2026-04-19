import cv2
from helper.config import CELL_CORRECTION_FACTOR

def drawFace(frame, possible_squares, id):
    face = []
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
                for square in face:
                    x, y, w, h = square["x"], square["y"], square["w"], square["h"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (220, 220, 220), 5)
                    cv2.putText(frame, f"{pos}", (x+int(0.5*w), y+int(0.5*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                    pos += 1

    return face, frame
