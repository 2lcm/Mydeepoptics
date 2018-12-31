import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from inference import *

# in tensorflow
# make PSF
# convolution
# add noise
# deconvolution
# calculate loss
aperture_diameter = 5e-3
distance = 35.5e-3
refractive_idcs = np.array([1.4648, 1.4599, 1.4568])
wave_lenghts = np.array([460, 550, 640]) * 1e-9
wave_resolution = 1024

def main():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    fig = plt.figure(figsize=(15, 15))

    # prepare variables
    img = load_image()
    img = tf.convert_to_tensor(img)
    h, w, _ch = map(int, img.shape)
    tf_img = tf.reshape(img, [1, h, w, _ch])

    h_map = tf.Variable(tf.ones([1, wave_resolution, wave_resolution, 1]))
    phase_shifts = phaseshifts_from_height_map(h_map, wave_lenghts, refractive_idcs)
    # fields = tf.ones([wave_resolution, wave_resolution])
    # original field ???
    # tf.multiply()

    psf = None
    kl = 11  # kernel length
    for i in range(_ch):
        psf_ch = None
        for j in range(_ch):
            if i==j:
                tmp = gkern(kl, 1.5)
                tmp = tf.reshape(tmp, [kl, kl, 1])
            else:
                tmp = tf.zeros([kl, kl, 1])
            if j==0:
                psf_ch = tmp
            else:
                psf_ch = tf.concat([psf_ch, tmp], 2)
        psf_ch = tf.reshape(psf_ch, [kl, kl, _ch, 1])
        if i==0:
            psf = psf_ch
        else:
            psf = tf.concat([psf, psf_ch], 3)

    # show original image
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("Original Image")
    plt_imshow(tf_img, sess)

    conv_img = tf.nn.conv2d(tf_img, psf, [1, 1, 1, 1], "SAME")
    conv_img = tf.clip_by_value(conv_img, 0, 1)

    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("convolution image, no noise")
    plt_imshow(conv_img, sess)

    # add noise
    noise = tf.random_normal([1, h, w, _ch], 0, 0.005)
    conv_img_noise = tf.add(conv_img, noise)
    conv_img_noise = tf.clip_by_value(conv_img_noise, 0, 1)

    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("convolution image with noise")
    plt_imshow(conv_img_noise, sess)

    deconv_tmp = inverse_filter(conv_img_noise, conv_img_noise, psf, 0.005)
    deconv = tf.zeros([h, w, 1])
    for ch in range(_ch):
        tmp = deconv_tmp[ch,:,:,ch]
        tmp = tf.reshape(tmp, [h, w, 1])
        deconv = tf.concat([deconv, tmp], 2)
    deconv = tf.reshape(deconv[:,:,1:4], [1, h, w, _ch])
    deconv = tf.clip_by_value(deconv, 0, 1)

    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("deconvolution image")
    plt_imshow(deconv, sess)

    # show all
    plt.show()

    loss = tf.norm(deconv - tf_img)
    print(loss.eval(session=sess))
    # print(loss)

main()
