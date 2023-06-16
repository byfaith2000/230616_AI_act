import cv2
import numpy as np
from matplotlib import pyplot as plt

def otsus_thresholding():
    img = cv2.imread('woman.jpeg', cv2.IMREAD_GRAYSCALE)
    th_max = 255

    ret_1, thresh_1 = cv2.threshold(img, 110, th_max, cv2.THRESH_BINARY)
    ret_2, thresh_2 = cv2.threshold(img, 0, th_max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    ret3, thresh_3 = cv2.threshold(blur, 0, th_max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    titles = ['ORIGINAL NOISY IMAGE', 'HISTOGRAM', 'GLOBAL THRESHOLDING(v=110)', 'ORIGINAL NOISY IMAGE', 'HISTOGRAM', "OTSU's THRESHOLIDNG", 'GAUSSIAN FILTERED IMAGE', 'HISTOGRAM', "OTSU's THRESHOLDING"]
    images = [img, 0, thresh_1, img, 0, thresh_2, blur, 0, thresh_3]

    fig = plt.figure()
    fig.canvas.set_window_title("JetsonNano_Thresholding")
    for i in range(3):
        plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()


otsus_thresholding()