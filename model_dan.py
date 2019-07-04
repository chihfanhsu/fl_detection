import tensorflow as tf
import numpy as np
import TensorflowUtils as utils

IMAGE_SIZE = 224

def batch_norm(x, train_phase, name='bn_layer'):
    #with tf.variable_scope(name) as scope:
    batch_norm = tf.layers.batch_normalization(
            inputs=x,
            momentum=0.9, epsilon=1e-5,
            center=True, scale=True,
            training = train_phase,
            name=name
    )
    return batch_norm

def conv_blk (inputs,n_filter, train_phase, name = 'conv_blk'):
    with tf.variable_scope(name):
        c1 = tf.layers.conv2d(inputs, filters=n_filter[0], kernel_size=[3,3], strides=(1,1), padding='same')       
        c1_bn = batch_norm(c1, train_phase, name='c1_bn')
        c1_relu = tf.nn.relu(c1_bn)
        c2 = tf.layers.conv2d(c1_relu,filters=n_filter[1],kernel_size=[3,3],strides=(1,1),padding='same')        
        c2_bn = batch_norm(c2, train_phase, name='c2_bn')
        c2_relu = tf.nn.relu(c2_bn)
        return c2_relu

def FF_NN(inputs, train_phase, keeprate):
    h1 = conv_blk(inputs, [64,64], train_phase, name='conv_blk1')
    m1 = tf.layers.max_pooling2d(h1,pool_size=[2,2],strides=(2,2))

    h2 = conv_blk(m1, [128,128], train_phase, name='conv_blk2')
    m2 = tf.layers.max_pooling2d(h2,pool_size=[2,2],strides=(2,2))

    h3 = conv_blk(m2, [256,256], train_phase, name='conv_blk3')
    m3 = tf.layers.max_pooling2d(h3,pool_size=[2,2],strides=(2,2))

    h4 = conv_blk(m3, [512,512], train_phase, name='conv_blk4')
    m4 = tf.layers.max_pooling2d(h4,pool_size=[2,2],strides=(2,2))

    flt = tf.layers.flatten(m4)

    # fully connected part
    f1_do = tf.layers.dropout(flt,rate=keeprate)
    f1 = tf.layers.dense(f1_do,256,activation=None)
    f1_bn = batch_norm(f1, train_phase, name='f1_bn')
    f1_relu = tf.nn.relu(f1_bn)
    
    f2 = tf.layers.dense(f1_relu,136,activation=None)
    y_out = tf.reshape(f2, shape=[-1,68,2])
    
    return y_out, f1_relu

#https://github.com/zjjMaiMai/Deep-Alignment-Network-A-convolutional-neural-network-for-robust-face-alignment/blob/master/DAN_V2/dan_model.py
# "shape" means points
def __calc_affine_params(from_shape,to_shape):
    from_shape = tf.cast(from_shape,dtype=tf.float32)    
    to_shape = tf.cast(to_shape,dtype=tf.float32)
    from_shape = tf.reshape(from_shape,[-1,68,2])
    to_shape = tf.reshape(to_shape,[-1,68,2])

    from_mean = tf.reduce_mean(from_shape, axis=1, keepdims=True)
    to_mean = tf.reduce_mean(to_shape, axis=1, keepdims=True)

    from_centralized = from_shape - from_mean
    to_centralized = to_shape - to_mean

    dot_result = tf.reduce_sum(tf.multiply(from_centralized, to_centralized), axis=[1, 2])
    norm_pow_2 = tf.pow(tf.norm(from_centralized, axis=[1, 2]), 2)

    a = dot_result / norm_pow_2
    b = tf.reduce_sum(tf.multiply(from_centralized[:, :, 0], to_centralized[:, :, 1]) - tf.multiply(from_centralized[:, :, 1], to_centralized[:, :, 0]), 1) / norm_pow_2

    r = tf.reshape(tf.stack([a, b, -b, a], axis=1), [-1, 2, 2])
    t = to_mean - tf.matmul(from_mean, r)
    return r,t

def __affine_image(imgs,r,t):
    # The Tensor [imgs].format is [NHWC]
    r = tf.matrix_inverse(r)
    r = tf.matrix_transpose(r)

    rm = tf.reshape(tf.pad(r, [[0, 0], [0, 0], [0, 1]], mode='CONSTANT'), [-1, 6])
    rm = tf.pad(rm, [[0, 0], [0, 2]], mode='CONSTANT')

    tm = tf.contrib.image.translations_to_projective_transforms(tf.reshape(t, [-1, 2]))
    rtm = tf.contrib.image.compose_transforms(rm, tm)
    
    # crash with GPU
    with tf.device('/cpu:0'):
        ret = tf.contrib.image.transform(imgs, rtm, interpolation="BILINEAR")

    return ret

def __affine_shape(shapes,r,t,isinv=False):
    if isinv:
        r = tf.matrix_inverse(r)
        t = tf.matmul(-t,r)
    shapes = tf.matmul(shapes,r) + t
    return shapes
def __gen_heatmap(shapes, IMAGE_SIZE=224):
    __pixels__ = tf.constant([(x, y) for y in range(IMAGE_SIZE) for x in range(IMAGE_SIZE)],
                                      dtype=tf.float32,shape=[1,IMAGE_SIZE,IMAGE_SIZE,2])
    shapes = shapes[:,:,tf.newaxis,tf.newaxis,:]
#     __pixels__ (1, 224, 224, 2)
#     shapes (?, 68, 1, 1, 2)
    value = __pixels__ - shapes
#   value (?, 68, 224, 224, 2)
    value = tf.norm(value,axis=-1)
#   value2 (?, 68, 224, 224)
    value = 1.0 / (tf.reduce_min(value,axis=1) + 1.0)
#   value3 (?, 224, 224)
    value = tf.expand_dims(value,axis=-1)
#   value_o (?, 224, 224, 1)
    return value

def conn_layers(imgs, mean_s, s_current, fc_current):   
    # Transformation estimation
    r,t = __calc_affine_params(s_current, mean_s)
    
    # image transformation
    T_img = __affine_image(imgs, r, t)
    
    # landmark transformation
    T_pts=__affine_shape(s_current, r, t, isinv=False)
    
    # heatmap generation
    hmap = __gen_heatmap(T_pts, IMAGE_SIZE//2)
    
    # feature generation
    fm_flat = tf.layers.dense(fc_current,(IMAGE_SIZE // 4) ** 2,activation=tf.nn.relu)
#     print('fm_flat', fm_flat)
    fm = tf.reshape(fm_flat, shape = [-1,(IMAGE_SIZE // 4),(IMAGE_SIZE // 4), 1])
    fmap = tf.image.resize_images(fm, (IMAGE_SIZE//2, IMAGE_SIZE//2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return T_img, hmap, fmap, T_pts, r, t

def DAN_blk(imgs, s_mean, s_current, fc_current, train_phase, keeprate, name = 'DAN_blk'):
    with tf.variable_scope(name):
        T_img, hmap, fmap, T_pts, r, t = conn_layers(imgs, s_mean, s_current, fc_current)
#         print('T_img', T_img)
#         print('hmap', hmap)
#         print('fmap', fmap)
        igt_input = tf.concat([T_img, hmap, fmap], axis=3)
    
        delta_s, fc_next = FF_NN(igt_input, train_phase, keeprate)

        s_next = T_pts + delta_s

        s_next_inverse = __affine_shape(s_next, r, t, isinv=True)

        return s_next_inverse, fc_next

def inference (image, s_mean, keep_prob, train_phase, debug):
    # DAN model

    imgs_in_1 = tf.image.resize_images(image, (IMAGE_SIZE//2, IMAGE_SIZE//2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    imgs_in_2 = tf.reduce_mean(imgs_in_1, axis=3, keepdims=True)
    
    delta_s, fc1 = FF_NN(imgs_in_2, train_phase, keep_prob)
    s1_out = tf.cast(s_mean,dtype=tf.float32) + delta_s

    # DAN block
    ds2, fc2 = DAN_blk(imgs_in_2, s_mean, s1_out, fc1, train_phase, keep_prob, name = 'DAN_blk1')
    ds3, fc3 = DAN_blk(imgs_in_2, s_mean, ds2, fc2, train_phase, keep_prob, name = 'DAN_blk2')
    ds4, fc4 = DAN_blk(imgs_in_2, s_mean, ds3, fc3, train_phase, keep_prob, name = 'DAN_blk3')
#     ds5, fc5 = DAN_blk(imgs_in_2, s_mean, ds4, fc4, train_phase, keep_prob, name = 'DAN_blk4')
#     ds6, fc6 = DAN_blk(imgs_in_2, s_mean, ds5, fc5, train_phase, keep_prob, name = 'DAN_blk5')
#     ds7, fc7 = DAN_blk(imgs_in_2, s_mean, ds6, fc6, train_phase, keep_prob, name = 'DAN_blk6')
    s2_out, _ = DAN_blk(imgs_in_2, s_mean, ds4, fc4, train_phase, keep_prob, name = 'DAN_blk6')
    # print(s_out)
    # s_out_flatten = tf.layers.flatten(s_out)
    return s2_out