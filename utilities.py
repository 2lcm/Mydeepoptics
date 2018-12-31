import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def load_images():
    images = None
    for file in os.listdir("./data/"):
        img = Image.open("./data/"+file)
        img = np.array(img, dtype=np.float32)
        img = img / 255
        a, b, c = img.shape
        img = np.reshape(img, [1, a, b, c])
        if images is None:
            images = img
        else:
            images = np.concatenate([images, img], axis=0)
    return images


def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)


def plt_imshow(img, sess):
    _, h, w, ch = img.shape
    img = tf.reshape(img, [h, w, ch])
    plt.imshow(img.eval(session=sess))

