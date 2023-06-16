import cv2
import numpy as np

def rectangle():
    img = np.zeros((512, 512, 3), np.uint8)
    img.fill(255)
    img = cv2.rectangle(img, (100, 150), (300, 180), (0, 255, 0), 7)
    cv2.imshow("Jet_Rect", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

rectangle()