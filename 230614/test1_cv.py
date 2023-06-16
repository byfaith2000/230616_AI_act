import cv2

def basic_operation():
    img_1 = cv2.imread('jungle1.png', cv2.IMREAD_COLOR)
    img_2 = img_1.copy()
    img_3 = img_1.copy()

    shape = img_1.shape
    size = img_1.size
    dtype = img_1.dtype

    cv2.putText(img_1, 'shape: ' + str(shape), (20, shape[0]-100), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,0), 2)
    cv2.putText(img_1, 'size: ' + str(size), (20, shape[0]-75), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
    cv2.putText(img_1, 'dtype: ' + str(dtype), (20, shape[0]-50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
    cv2.imshow('JetsonNano_Basic_Operation', img_1)

    b,g,r = cv2.split(img_2)
    img_2 = cv2.merge((r,g,b))
    cv2.imshow('JetsonNano_Basic_Operation_splitNmerge', img_2)

    img_3[0:150,0:150,2] = 0
    cv2.imshow('JetsonNano_Basic_Operation_Channel', img_3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    basic_operation()

