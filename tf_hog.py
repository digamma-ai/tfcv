import numpy as np
import tensorflow as tf
from tf_filters import tf_deriv


def tf_hog_descriptor(images, cell_size = 8, block_size = 2, block_stride = 1, n_bins = 9,
                      grayscale = False, oriented = False):

    batch_size, height, width, depth = images.shape
    half_pi = tf.constant(np.pi/2, name="pi_half")
    eps = tf.constant(1e-6, name="eps")
    scale_factor = tf.constant(np.pi * n_bins * 0.99999, name="scale_factor")

    img_idx, row_idx, col_idx, _ = np.indices(images.shape)
    img_idx = tf.constant(img_idx, name="ImgIndex", dtype=tf.int32)
    row_idx = tf.constant(row_idx, name="RowIndex", dtype=tf.int32)
    col_idx = tf.constant(col_idx, name="ColIndex", dtype=tf.int32)
    
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
    max_idx = tf.argmax(g_norm, 3)
    max_idx = tf.to_int32(max_idx)
    max_idx = tf.pack([max_idx]*depth)
    max_idx = tf.transpose(max_idx, [1,2,3,0])
    idx = tf.pack([img_idx, row_idx, col_idx, max_idx])
    idx = tf.transpose(idx, [1,2,3,4,0], name="GradArgmax")

    g_norm = tf.gather_nd(g_norm, idx, name="MaxGradNorm")[:,:,:,:1] 
    g_x   = tf.gather_nd(g_x, idx, name="MaxGradX")[:,:,:,:1] 
    g_y   = tf.gather_nd(g_y, idx, name="MaxGradY")[:,:,:,:1]

    # orientation and binning
    if oriented:
        # atan2 implementation needed 
        # lots of conditional indexing required
        raise NotImplementedError("`oriented` gradient not supported yet")
    else:
        g_dir = tf.atan(g_y / (g_x + eps)) + half_pi
        g_bin = g_dir / scale_factor
        g_bin = tf.floor(g_bin)
        g_bin = tf.to_int32(g_bin, name="Bins")  

    # cells partitioning
    cell_norm = tf.space_to_depth(g_norm, cell_size, name="GradCells")
    cell_bins = tf.space_to_depth(g_bin, cell_size, name="BinsCells")

    # cells histograms
    hist = list()
    for i in range(n_bins):
        mask = tf.equal(cell_bins, tf.constant(i, name="%i"%i))
        mask = tf.to_float(mask)
        bin_values = mask * cell_norm
        bin_values = tf.reduce_sum(bin_values, 3)
        hist.append(bin_values)
    hist = tf.pack(hist)
    hist = tf.transpose(hist, [1,2,3,0], name="Hist")

    # blocks partitioning
    block_hist = tf.extract_image_patches(hist, 
                                          ksizes  = [1, block_size, block_size, 1], 
                                          strides = [1, block_stride, block_stride, 1], 
                                          rates   = [1,1,1,1], 
                                          padding = 'VALID',
                                          name    = "BlockHist")

    # block normalization
    norm_const = tf.square(block_hist)
    norm_const = tf.reduce_sum(norm_const, 3)
    norm_const = tf.sqrt(norm_const + 1)
    norm_const = tf.expand_dims(norm_const, -1)
    norm_const = tf.tile(norm_const, [1, 1, 1, n_bins*block_size*block_size], name='NormConst')
    block_hist = block_hist / norm_const
    
    # HOG descriptor
    hog_descriptor = tf.reshape(block_hist, 
                                [int(block_hist.get_shape()[0]), 
                                     int(block_hist.get_shape()[1]) * \
                                     int(block_hist.get_shape()[2]) * \
                                     int(block_hist.get_shape()[3])], 
                                name='HOGDescriptor')

    return hog_descriptor