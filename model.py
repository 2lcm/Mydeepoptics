import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from inference import *


class DeepModel(object):
    def __init__(self, shape):
        # constants
        self.kernel_length = 11
        self.aperture_diameter = 5e-3
        self.distance = 35.5e-3
        self.refractive_idcs = np.array([1.4648, 1.4599, 1.4568])
        self.wave_lenghts = np.array([460, 550, 640]) * 1e-9
        self.wave_resolution = 1024

        # (# of image, height, width, channel)
        self.x = tf.placeholder(tf.float32, shape)
        self.y = tf.placeholder(tf.float32, shape)
        self.h_map = tf.Variable(tf.ones([1, self.wave_resolution, self.wave_resolution, 1]))
        # self.psf = None
        # phaseshifts ...

        self.psf = tf.Variable(self.make_psf(self.kernel_length))
        self.fig = plt.figure(figsize=(15, 15))
        self.conv_img = tf.nn.conv2d(self.x, self.psf, [1, 1, 1, 1], "SAME")
        self.conv_img = tf.clip_by_value(self.conv_img, 0, 1)
        noise = tf.random_normal(tf.shape(self.conv_img), 0, 0.005)
        self.conv_img_noise = tf.add(self.conv_img, noise)
        self.conv_img_noise = tf.clip_by_value(self.conv_img_noise, 0, 1)
        # deconvolution
        deconv_tmp = inverse_filter(self.conv_img_noise, self.conv_img_noise, self.psf, 0.005)
        print(deconv_tmp)
        # for ch in range(chs):
        #     tmp = deconv_tmp[ch, :, :, ch]
        #     tmp = tf.reshape(tmp, [h, w, 1])
        #     deconv = tf.concat([deconv, tmp], 2)
        # deconv = tf.reshape(deconv[:, :, 1:4], [1, h, w, chs])
        # deconv = tf.clip_by_value(deconv, 0, 1)
        # deconv = tf.reshape(deconv[:, :, 1:4], [1, h, w, chs])
        # deconv = tf.clip_by_value(deconv, 0, 1)
        self.loss = tf.reduce_mean(tf.square(self.deconv - self.y))

    def run(self, images):
        opt_params = {'momentum': 0.5, 'use_nesterov': True}
        learning_rate = 0.001
        # adam
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, **opt_params)
        # sgd
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, **opt_params)
        op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        with tf.Session() as self.sess:
            self.sess.run(init)
            results, loss, _ = self.sess.run([self.deconv, self.loss, op], feed_dict={self.x: images, self.y: images})
            return results, loss


    def make_psf(self, kl):
        psf = np.zeros([kl, kl, 3, 3], np.float32)
        for i in range(3):
            tmp = gkern(kl, 2)
            # tmp = np.reshape(tmp, [kl, kl, 1, 1])
            psf[:,:,i,i] = tmp
        psf = tf.convert_to_tensor(psf, tf.float32)
        return psf

    def img_show(self, image, i, title):
        ax = self.fig.add_subplot(2, 2, i)
        ax.set_title(title)
        plt_imshow(image, self.sess)


if __name__ == "__main__":
    images = load_images()
    shape = images.shape
    model = DeepModel(shape)
    res, loss = model.run(images)

