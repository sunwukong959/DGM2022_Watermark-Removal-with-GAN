import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path):
    """
    The input image is loaded and converted to numpy array [-1, 1].

    :param path: the path of the input image
    :return: numpy array containing image
    """

    img = Image.open(path)
    img = np.asarray(img)
    img = (img-np.min(img)) * 2.0 / (np.max(img)-np.min(img)) - 1
    return img


if __name__ == '__main__':
    img = load_image('./data/raw/image_0001.png')
    print('image shape: ', img.shape)
    plt.imshow(img)
    plt.show()
