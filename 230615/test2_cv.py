import cv2
import numpy as np

def object_tracking():
    frame = cv2.imread('idol.jpg')
    frame = cv2.resize(frame, (640, 480))

    cv2.imshow("JetsonNano_OT_Original", frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("JetsonNano_OT_HSV", hsv)

    lower_red = np.array([0, 10, 10])
    upper_red = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    cv2.imshow("JetsonNano_OT_Mask", mask)

    res = cv2.bitwise_and(frame, frame, mask = mask)
    cv2.imshow("JetsonNano_OT_Result", res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    object_tracking()