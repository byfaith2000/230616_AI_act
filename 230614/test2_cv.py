import cv2

def nothing(x):
    pass

def image_blending():
    cv2.namedWindow('JetsonNano_Image_Blending')
    cv2.createTrackbar('w', 'JetsonNano_Image_Blending', 0, 100, nothing)

    img_1 = cv2.imread('monkey1.png')
    img_1 = cv2.resize(img_1, (512,512))
    img_2 = cv2.imread('jungle1.png')
    img_2 = cv2.resize(img_2, (512, 512))

    while (True):
        w = cv2.getTrackbarPos('w', 'JetsonNano_Image_Blending')
        dst = cv2.addWeighted(img_1, float(100-w)*0.01, img_2, float(w)*0.01, 0)
        cv2.imshow('JetsonNano_Image_Blending', dst)
        if cv2.waitKey(1) & 0xFF ==27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_blending()