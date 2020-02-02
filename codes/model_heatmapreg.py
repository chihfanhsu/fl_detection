import tensorflow as tf
import numpy as np
import TensorflowUtils as utils

NUM_OF_CLASSESS = int(68 + 1)

def batch_norm(x, train_phase, name='bn_layer'):
    batch_norm = tf.layers.batch_normalization(
            inputs=x,
            momentum=0.9, epsilon=1e-5,
            center=True, scale=True,
            training = train_phase,
            name=name
    )
    return batch_norm

def conv_blk (inputs, n_filter, kernel_size, train_phase, name = 'conv_blk'):
    with tf.variable_scope(name):
        c1 = tf.layers.conv2d(inputs, filters=n_filter, kernel_size=kernel_size, strides=(1,1), padding='same')       
        c1_bn = batch_norm(c1, train_phase, name='c1_bn')
        c1_relu = tf.nn.relu(c1_bn)
        return c1_relu

def inference(image, keep_prob, train_phase, debug):
    # model
    with tf.variable_scope("inference"):

        image1 = tf.reduce_mean(image, axis=3, keepdims=True)
        
        c1_1 = conv_blk(image1, 64, [3,3], train_phase, "c1_1")
        c1_2 = conv_blk(c1_1, 64, [3,3], train_phase, "c1_2")
        pool1 = utils.max_pool_2x2(c1_2); print("pool1",pool1)
        
        c2_1 = conv_blk(pool1, 128, [3,3], train_phase, "c2_1")
        c2_2 = conv_blk(c2_1, 128, [3,3], train_phase, "c2_2")
        pool2 = utils.max_pool_2x2(c2_2); print("pool2",pool2)
        
        c3_1 = conv_blk(pool2, 256, [3,3], train_phase, "c3_1")
        c3_2 = conv_blk(c3_1, 256, [3,3], train_phase, "c3_2")
        c3_3 = conv_blk(c3_2, 256, [3,3], train_phase, "c3_3")
        c3_4 = conv_blk(c3_3, 256, [3,3], train_phase, "c3_4")
        pool3 = utils.max_pool_2x2(c3_4); print("pool3",pool3)
        
        c4_1 = conv_blk(pool3, 512, [3,3], train_phase, "c4_1")
        c4_2 = conv_blk(c4_1, 512, [3,3], train_phase, "c4_2")
        c4_3 = conv_blk(c4_2, 512, [3,3], train_phase, "c4_3")
        c4_4 = conv_blk(c4_3, 512, [3,3], train_phase, "c4_4")
        pool4 = utils.max_pool_2x2(c4_4); print("pool4",pool4)
        
        c5_1 = conv_blk(pool4, 512, [3,3], train_phase, "c5_1")
        c5_2 = conv_blk(c5_1, 512, [3,3], train_phase, "c5_2")
        c5_3 = conv_blk(c5_2, 512, [3,3], train_phase, "c5_3")
        c5_4 = conv_blk(c5_3, 512, [3,3], train_phase, "c5_4")
        pool5 = utils.max_pool_2x2(c5_4); print("pool5",pool5)
        
        # conv layers
        c6_1 = conv_blk(pool5, 2048, [7,7], train_phase, "c6_1")
        c6_2 = conv_blk(c6_1, 2048, [1,1], train_phase, "c6_2")

        conv8=tf.contrib.layers.conv2d(c6_2, NUM_OF_CLASSESS, [1,1], stride=1,padding='SAME')

        # now to upscale to actual image size
        deconv_shape1 = pool4.get_shape()
        conv_t1 = tf.contrib.layers.conv2d_transpose(conv8,
                                                     deconv_shape1[3].value,
                                                     [4, 4],
                                                     stride=2,
                                                     padding='SAME',
                                                     activation_fn=None)
        fuse_1 = tf.add(conv_t1, pool4, name="fuse_1");print("fuse_1",fuse_1)

        deconv_shape2 = pool3.get_shape()
        conv_t2 = tf.contrib.layers.conv2d_transpose(fuse_1,
                                                     deconv_shape2[3].value,
                                                     [4, 4],
                                                     stride=2,
                                                     padding='SAME',
                                                     activation_fn=None)
        fuse_2 = tf.add(conv_t2, pool3, name="fuse_2");print("fuse_2",fuse_2)

        fuse_3 = tf.contrib.layers.conv2d_transpose(fuse_2,
                                                     NUM_OF_CLASSESS,
                                                     [16, 16],
                                                     stride=8,
                                                     padding='SAME',
                                                     activation_fn=None);print("fuse_3",fuse_3)

        return fuse_3