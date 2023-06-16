import cv2
import numpy as np

def text():
    img = np.zeros((512, 512, 3), np.uint8)
    img.fill(255)
    img = cv2.rectangle(img, (100, 150), (500, 300), (0, 255, 0), 7)
    img = cv2.putText(img, 'Hello', (100, 150), cv2.FONT_HERSHEY_PLAIN, 6, (0, 100, 256), 5)
    img = cv2.putText(img, 'Hello', (100, 220), cv2.FONT_HERSHEY_PLAIN, 6, (0, 100, 256), 5)
    img = cv2.line(img, (300, 0), (100, 511), (0, 255, 0), 5)
    img = cv2.ellipse(img, (256, 256), (100, 200), 45, -30, 330, (156, 50, 230), 5)
 

    cv2.imshow("Jet_Rect", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
  text()

                      