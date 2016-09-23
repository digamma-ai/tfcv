import numpy as np
import tensorflow as tf
from tf_filters import tf_deriv


def tf_select_by_idx(a, idx):
    return tf.select(tf.equal(idx, 2), 
                     a[:,:,:,2], 
                     tf.select(tf.equal(idx, 1), 
                               a[:,:,:,1], 
                               a[:,:,:,0]))


def tf_hog_descriptor(images, cell_size = 8, block_size = 2, block_stride = 1, n_bins = 9,
                      grayscale = False, oriented = False):

    batch_size, height, width, depth = images.shape
    half_pi = tf.constant(np.pi/2, name="pi_half")
    eps = tf.constant(1e-6, name="eps")
    scale_factor = tf.constant(np.pi * n_bins * 0.99999, name="scale_factor")
    
    img = tf.constant(images, name="ImgBatch", dtype=tf.float32)

    # gradients
    if grayscale:
        gray = tf.image.rgb_to_grayscale(img, name="ImgGray")
        grad = tf_deriv(gray)
    else:
        grad = tf_deriv(img)
    g_x = grad[:,:,:,0::2]
    g_y = grad[:,:,:,1::2]
    
    # maximum norm gradient selection
    g_norm = tf.sqrt(tf.square(g_x) + tf.square(g_y), "GradNorm")
    idx    = tf.argmax(g_norm, 3)
    
    g_norm = tf.expand_dims(tf_select_by_idx(g_norm, idx), -1)
    g_x    = tf.expand_dims(tf_select_by_idx(g_x,    idx), -1)
    g_y    = tf.expand_dims(tf_select_by_idx(g_y,    idx), -1)

    # orientation and binning
    if oriented:
        # atan2 implementation needed 
        # lots of conditional indexing required
        raise NotImplementedError("`oriented` gradient not supported yet")
    else:
        g_dir = tf.atan(g_y / (g_x + eps)) + half_pi
        g_bin = tf.to_int32(g_dir / scale_factor, name="Bins")  

    # cells partitioning
    cell_norm = tf.space_to_depth(g_norm, cell_size, name="GradCells")
    cell_bins = tf.space_to_depth(g_bin,  cell_size, name="BinsCells")

    # cells histograms
    hist = list()
    zero = tf.zeros(cell_bins.get_shape()) 
    for i in range(n_bins):
        mask = tf.equal(cell_bins, tf.constant(i, name="%i"%i))
        hist.append(tf.reduce_sum(tf.select(mask, cell_norm, zero), 3))
    hist = tf.transpose(tf.pack(hist), [1,2,3,0], name="Hist")

    # blocks partitioning
    block_hist = tf.extract_image_patches(hist, 
                                          ksizes  = [1, block_size, block_size, 1], 
                                          strides = [1, block_stride, block_stride, 1], 
                                          rates   = [1, 1, 1, 1], 
                                          padding = 'VALID',
                                          name    = "BlockHist")

    # block normalization
    block_hist = tf.nn.l2_normalize(block_hist, 3, epsilon=1.0)
    
    # HOG descriptor
    hog_descriptor = tf.reshape(block_hist, 
                                [int(block_hist.get_shape()[0]), 
                                     int(block_hist.get_shape()[1]) * \
                                     int(block_hist.get_shape()[2]) * \
                                     int(block_hist.get_shape()[3])], 
                                 name='HOGDescriptor')

    return hog_descriptor