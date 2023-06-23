import cv2
import numpy as np

def image_contours():

    img = cv2.imread('idol.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img_thresh = cv2.threshold(img_gray, 249, 255, 0)
    contours, hierachy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = img.copy()
    img_contours = cv2.drawContours(img_contours, contours, 3, (51, 0, 255), 10)

    cv2.imshow('JetsonNano_Contours_Original', img)
    cv2.imshow('JetsonNano_Contours_Gray', img_gray)
    cv2.imshow('JetsonNano_Contours_Thresh', img_thresh)
    cv2.imshow('JetsonNano_Contours', img_contours)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_contours()