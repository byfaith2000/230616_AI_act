import cv2
from matplotlib import pyplot as plt

def matplotlib():
    img = cv2.imread('/home/nano/Desktop/LSTM/rgb_header.png', cv2.IMREAD_COLOR)

    fig = plt.figure()
    fig.canvas.set_window_title('JetsonNano_matplot')
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    fig = plt.figure()
    fig.canvas.set_window_title('JetsonNano_')
    b,g,r = cv2.split(img)
    img2 = cv2.merge([r,g,b])
    plt.imshow(img2)
    plt.xticks([])
    plt.yticks([])
    plt.show()



matplotlib()