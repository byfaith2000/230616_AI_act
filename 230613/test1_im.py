import cv2
import numpy as np
from math import sqrt, pow

mode = 'char' #initial value
drawing = False
ix, iy = -1, -1
img = np.zeros((512, 512, 3), np.uint8)+255
img.fill(255)  # 

def draw_figure(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == 'r':
                cv2.rectangle(img, (ix, iy), (x, y),
                              (255, 0, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == 'r':
            cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 255), -1)
        elif mode == 'l':
             cv2.line(img, (ix, iy), (x, y), (255, 0, 255), 2)
        elif mode == 'c':
             radius = int(sqrt(pow(x-ix, 2) + pow(y-iy, 2)) / 2)
             cv2.circle(img, (x, y), radius, (0, 9, 255), -1)

def mouse_draw_shape():
    global img, mode
    cv2.namedWindow('JetsonNano_MouseDrawing')
    cv2.setMouseCallback('JetsonNano_MouseDrawing', draw_figure)

    while True:
        cv2.imshow('JetsonNano_MouseDrawing', img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            mode = 'r'
        elif key == ord('c'):
            mode = 'c'
        elif key == ord('l'):
            mode = 'l'
        elif key == ord('e'):
            img = np.zeros((512, 512, 3), np.uint8)
        elif key == 27:
            break
        else:
            continue

cv2.destroyAllWindows()

if __name__ == "__main__":
    mouse_draw_shape()

