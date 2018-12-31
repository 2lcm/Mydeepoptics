# source: https://github.com/vsitzmann/deepoptics/blob/master/src/layers/deconv.py

import tensorflow as tf
import numpy as np

def ifftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
       n = input_shape[axis]
       split = n - (n + 1) // 2
       mylist = np.concatenate((np.arange(split, n), np.arange(split)))
       new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor

def psf2otf(input_filter, output_size):
    """Convert 4D tensorflow filter into its FFT.
    """
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    fh, fw, _, _ = input_filter.shape.as_list()

    if output_size[0] != fh:
       pad1 = (output_size[0] - fh)/2
       pad2 = (output_size[1] - fh) / 2

       if (output_size[0] - fh) % 2 != 0:
          pad_top = int(np.ceil(pad1))
          pad_bottom = int(np.floor(pad1))
       else:
          pad_top = int(pad1) + 1
          pad_bottom = int(pad1) - 1

       if (output_size[1] - fh) % 2 != 0:
          pad_left= int(np.ceil(pad2))
          pad_right = int(np.floor(pad2))
       else:
          pad_left = int(pad2) + 1
          pad_right = int(pad2) - 1

       padded = tf.pad(input_filter, [[pad_top, pad_bottom],
                               [pad_left, pad_right], [0,0], [0,0]], "CONSTANT")
    else:
       padded = input_filter

    padded = tf.transpose(padded, [2,0,1,3])
    padded = ifftshift2d_tf(padded)
    padded = tf.transpose(padded, [1,2,0,3])

    ## Take FFT
    tmp = tf.transpose(padded, [2,3,0,1])
    tmp = tf.fft2d(tf.complex(tmp, 0.))
    return tf.transpose(tmp, [2,3,0,1])


def inverse_filter(blurred, estimate, psf, gamma=None, otf=None, init_gamma=1.5):
    """Implements Weiner deconvolution in the frequency domain, with circular boundary conditions.
    Args:
        blurred: image with shape (batch_size, height, width, num_img_channels)
        estimate: image with shape (batch_size, height, width, num_img_channels)
        psf: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)
    TODO precompute OTF, adj_filt_img.
    """
    img_shape = blurred.shape.as_list()

    if gamma is None:
        gamma_initializer = tf.constant_initializer(init_gamma)
        gamma = tf.get_variable(name="wiener_gamma",
                                shape=(),
                                dtype=tf.float32,
                                trainable=True,
                                initializer=gamma_initializer)
        gamma = tf.square(gamma)
        tf.summary.scalar('gamma', gamma)

    a_tensor_transp = tf.transpose(blurred, [0, 3, 1, 2])
    estimate_transp = tf.transpose(estimate, [0, 3, 1, 2])
    # Everything has shape (batch_size, num_channels, height, width)
    img_fft = tf.fft2d(tf.complex(a_tensor_transp, 0.))
    if otf is None:
        otf = psf2otf(psf, output_size=img_shape[1:3])
        otf = tf.transpose(otf, [2, 3, 0, 1])
    # otf: [input_channels, output_channels, height, width]
    # img_fft: [batch_size, channels, height, width]
    adj_conv = img_fft * tf.conj(otf)
    numerator = adj_conv + tf.fft2d(tf.complex(gamma * estimate_transp, 0.))

    kernel_mags = tf.square(tf.abs(otf))

    denominator = tf.complex(kernel_mags + gamma, 0.0)
    filtered = tf.div(numerator, denominator)
    cplx_result = tf.ifft2d(filtered)
    real_result = tf.real(cplx_result)
    # Get back to (batch_size, num_channels, height, width)
    result = tf.transpose(real_result, [0, 2, 3, 1])
    return result


def phaseshifts_from_height_map(height_map, wave_lengths, refractive_idcs):
    '''Calculates the phase shifts created by a height map with certain
    refractive index for light with specific wave length.
    '''
    # refractive index difference
    delta_N = refractive_idcs.reshape([1,1,1,-1]) - 1.
    # wave number
    wave_nos = 2. * np.pi / wave_lengths
    wave_nos = wave_nos.reshape([1,1,1,-1])
    # phase delay indiced by height field
    phi = wave_nos * delta_N * height_map
    phase_shifts = compl_exp_tf(phi)
    return phase_shifts


def compl_exp_tf(phase, dtype=tf.complex64, name='complex_exp'):
    """Complex exponent via euler's formula, since Cuda doesn't have a GPU kernel for that.
    Casts to *dtype*.
    """
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype),
                  1.j * tf.cast(tf.sin(phase), dtype=dtype),
                  name=name)

