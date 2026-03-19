import cv2
from opencv import extract_face

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.rectangle(color, (5, 5), (210, 210), (255, 0, 0), 2)
    cv2.rectangle(color, (5,5), (140,210), (255, 0, 0), 2)
    cv2.rectangle(color, (5, 5), (70, 210), (255, 0, 0), 2)
    cv2.rectangle(color, (5,5), (210,140), (255, 0, 0), 2)
    cv2.rectangle(color, (5, 5), (210, 70), (255, 0, 0), 2)


    cv2.imshow('frame', color)
    key_press = cv2.waitKey(1) & 0xFF

    faces = []

    if key_press == ord('q'):
        break
    elif key_press == ord('c'):
        faces.append(extract_face(frame=color))


cap.release()
cv2.destroyAllWindows()