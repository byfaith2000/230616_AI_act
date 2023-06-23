import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_rgb():

    img1 = cv2.imread('mountain1.jpg', cv2.IMREAD_COLOR)

    b,g,r = cv2.split(img1)

    img_1 = cv2.merge([r,g,b])
    be = cv2.equalizeHist(b)
    ge = cv2.equalizeHist(g)
    re = cv2.equalizeHist(r)
    img_2 = cv2.merge([re,ge,be])

    # cv2.imshow(img_1)
    # cv2.imshow(img_2)
    cv2.imshow('JetsonNano_Contours_Thresh', img_1)
    cv2.imshow('JetsonNano_Contours', img_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # def cvt(i):
    #     b,g,r = cv2.split(i)
    #     i = cv2.merge([r,g,b])
    #     return i
    # img1 = cvt(img1)
    #img2 = cvt(img2)

    #hist1 = cv2.calcHist([img1],[2],None, [256],[0,256])


    # fig = plt.figure()
    # fig.canvas.set_window_title('JetsonNano_Histo_rgb')

    # plt.subplot(121), plt.imshow(img_1, 'gray1')
    # plt.subplot(122), plt.imshow(img_2, 'gray2')

   # plt.subplot(223), plt.plot(hist1), plt.xlim([0,256])





    #plt.show()




histogram_rgb()