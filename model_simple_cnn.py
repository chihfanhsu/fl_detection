import tensorflow as tf
import numpy as np
import TensorflowUtils as utils

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

def inference(image, s_mean, keep_prob, train_phase, debug):
	r1 = tf.layers.max_pooling2d(image,pool_size=[2,2],strides=(2,2))

	h1 = conv_blk(r1, [64,64], train_phase, name='conv_blk1')
	m1 = tf.layers.max_pooling2d(h1,pool_size=[2,2],strides=(2,2))

	h2 = conv_blk(m1, [128,128], train_phase, name='conv_blk2')
	m2 = tf.layers.max_pooling2d(h2,pool_size=[2,2],strides=(2,2))

	h3 = conv_blk(m2, [256,256], train_phase, name='conv_blk3')
	m3 = tf.layers.max_pooling2d(h3,pool_size=[2,2],strides=(2,2))

	h4 = conv_blk(m3, [512,512], train_phase, name='conv_blk4')
	m4 = tf.layers.max_pooling2d(h4,pool_size=[2,2],strides=(2,2))

	flt = tf.layers.flatten(m4)

	# fully connected part

	f1_do = tf.layers.dropout(flt, rate=keep_prob)

	f1 = tf.layers.dense(f1_do,256,activation=None)
	f1_bn = batch_norm(f1, train_phase, name='f1_bn')
	f1_relu = tf.nn.relu(f1_bn)

	f2 = tf.layers.dense(f1_relu,136,activation=None)

	y_out = tf.reshape(f2, shape = [-1,68,2])

	return y_out