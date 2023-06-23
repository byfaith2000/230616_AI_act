import cv2
import numpy as np
from matplotlib import pyplot as plt

def image_gradient():

    img = cv2.imread("idol.jpg")
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    laplacian = cv2.Laplacian(img, cv2.CV_8U)  #CV_8U is gray..
    sobelx = cv2.Sobel(img, cv2.CV_8U,1,0,ksize=3) #mibun to X axis, 
    sobely = cv2.Sobel(img, cv2.CV_8U,1,1,ksize=3)
    canny = cv2.Canny(img, 30, 360)

    images = [img,laplacian,sobelx,sobely,canny]
    titles = ["Original",'Laplacian', 'Sobel_x', 'Sobel_y', 'Canny']

    fig = plt.figure()
    fig.canvas.set_window_title("JetsonNano_Image_Gradient")
    for i in range(2):
        plt.subplot(2,2,i+1),plt.imshow(images[i]), plt.title([titles[i]])
        plt.xticks([]), plt.yticks([])
    for i in range(3):
        plt.subplot(2,3,i+4),plt.imshow(images[i+2]), plt.title([titles[i+2]])
        plt.xticks([]), plt.yticks([])

    plt.show()


image_gradient()
