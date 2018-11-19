from random import random
import numpy as np
from skimage.transform import AffineTransform, warp
from PIL import Image


class ShiftTransform(object):
    def __init__(self, x, y):
        """
        :param x(float): fraction of total width, 0 < x < 1.0
        :param y(float): fraction of total height, 0 < y < 1.0
        """
        super(ShiftTransform, self).__init__()
        self.x = x
        self.y = y

    def __call__(self, img):
        """
        :param img: PIL Image
        :return: PIL Image
        """
        x = int((random() - 0.5) / 0.5 * self.x * img.size[0])
        y = int((random() - 0.5) / 0.5 * self.y * img.size[1])
        tmp_img = np.array(img)
        transform = AffineTransform(translation=(x, y))
        shifted_img = warp(tmp_img, transform, mode="edge", preserve_range=True)
        shifted_img = shifted_img.astype(tmp_img.dtype)
        return Image.fromarray(shifted_img)

    def __repr__(self):
        return self.__class__.__name__ + "(x: {}, y: {})".format(self.x, self.y)