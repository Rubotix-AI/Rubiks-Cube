import cv2
import numpy as np

from feed.processing import findFace
from solver import solve

cube_mat = np.zeros((6, 3, 3), dtype='uint8')

feed = cv2.VideoCapture(2) # External Webcam
if not feed.isOpened():
    print("Cannot Open Camera")
    exit()

faceID = 0
cube = []

while True and len(cube) < 7 and faceID < 54:
    read, frame = feed.read()

    if not read:
        print("Cannot recieve feed from camera.")
        break

    key = cv2.waitKey(20)
    
    face_info, processed_frame = findFace(frame=frame, id=faceID)

    cv2.imshow('frame', processed_frame)

    if key == ord('c'):
        cube.append(face_info)
        faceID += 9

    if key == ord('q'):
        print("Exiting feed read.")
        break

if len(cube) == 6:
    moves = solve(cube)
    print(moves)

feed.release()
cv2.destroyAllWindows()