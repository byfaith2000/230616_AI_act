import cv2
import numpy as np
from matplotlib import pyplot as plt

def image_adaptive_thresholding():
    img = cv2.imread('grad.jpeg', cv2.IMREAD_GRAYSCALE)
    th = 127
    th_max = 255

    ret1, thresh_1 = cv2.threshold(img, th, th_max, cv2.THRESH_BINARY)
    str_1 = 'GLOBAL'
    thresh_2 = cv2.adaptiveThreshold(img, th_max, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    str_2 = 'MEAN'
    thresh_3 = cv2.adaptiveThreshold(img, th_max, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    str_3 = 'GAUSSIAN'

    titles = ['ORIGINAL', str_1, str_2, str_3]
    images = [img, thresh_1, thresh_2, thresh_3]

    fig = plt.figure()
    fig.canvas.set_window_title("JetsonNano_Thresholding")
    for i in range(4):
        plt.subplot(2,2,i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


image_adaptive_thresholding()

