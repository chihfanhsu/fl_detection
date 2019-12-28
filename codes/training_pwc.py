
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import os, sys
import TensorflowUtils as utils
import pickle
import cv2
import pandas
import config
import tqdm
import time
conf, _ = config.get_config()


# # Set visiable GPU

# In[ ]:


'''setting'''
gpus = [conf.gpu] # Here I set CUDA to only see one GPU
os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(i) for i in gpus])


# # Set parameters

# In[ ]:


if conf.tar_model == 'pwc':
    import model_pwc as model
else:
    sys.exit("Sorry, Wrong Model!")

# load model
model_dir = './' + conf.tar_model + '/'
data_dir = './dataset/'
logs_dir = model_dir+'logs/'
imgs_dir = model_dir+'imgs/'
pred_dir = model_dir+'pred/'
NUM_OF_CLASSESS = int(68 + 1)
IMAGE_SIZE = 224

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)


# # Set model

# In[ ]:


db_helen = pickle.load(open(data_dir+conf.dataset+".pickle", "rb" ) )
# print the data structure
print(db_helen.keys())
print(db_helen['trainset'].keys())
# print the shape of tratining set
print(db_helen['trainset']['pts'].shape)
print(db_helen['trainset']['img'].shape)
# print the shape of testing set
print(db_helen['testset']['pts'].shape)
print(db_helen['testset']['img'].shape)
# declear data iterator
train_batches = utils.get_batch(db_helen['trainset']['img'], db_helen['trainset']['pts'], batch_size = conf.batch_size)
valid_batches = utils.get_batch(db_helen['testset']['img'], db_helen['testset']['pts'], batch_size = conf.batch_size)


# In[ ]:


tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default() as g:
    # model input
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    train_phase = tf.placeholder(tf.bool, name='phase_train')

    # model structure
    pred_annotation, logits = model.inference(image, keep_probability, train_phase, conf.debug)
    
    heat_map = tf.nn.softmax(logits)

    # check # of parameter
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('total_parameters', total_parameters)

    # model loss
    a = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                       name="entropy")
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)

    def train(loss_val, var_list):
        optimizer = tf.train.AdamOptimizer(conf.lr)
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        return optimizer.apply_gradients(grads)

    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=g)
    if (conf.training == False):
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading sucessfully')
        else:
            print('No checkpoint file found')
            raise
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    train_writer = tf.summary.FileWriter(logs_dir+'/train', sess.graph)
    validation_writer = tf.summary.FileWriter(logs_dir+'/validation', sess.graph)


# In[ ]:


# training
if conf.training == True:
    stop_count = 0
    itr = 0
    max_validloss = 99999
    while True:
        # prepare training input
        batch_xs, batch_ys = next(train_batches)
        batch_xs_aug, batch_ys_aug = utils.data_augmentation(batch_xs, batch_ys)
        batch_ymap_aug = utils.pts2map(batch_ys_aug)

        feed_dict = {image: batch_xs_aug/255,
                     annotation: batch_ymap_aug,
                     keep_probability: 0.5,
                     train_phase:conf.training}
        sess.run([train_op,extra_update_ops], feed_dict=feed_dict)

        if itr % 500 == 0:
            train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
            print("[T] Step: %d, loss:%g" % (itr, np.mean(train_loss)))
            train_writer.add_summary(summary_str, itr)
        # validation
        if itr % 1000 == 0:
            # prepare inputs
            valid_losses = []
            for i in tqdm.trange(int((db_helen['testset']['pts'].shape[0])/conf.batch_size)):
                batch_xs_valid, batch_ys_valid = next(valid_batches)
                batch_ymap_valid = utils.pts2map(batch_ys_valid);# print(batch_ymap_valid.shape)

                feed_dict = {image: batch_xs_valid/255,
                             annotation: batch_ymap_valid,
                             keep_probability: 1.0,
                             train_phase:conf.training}
                valid_loss, pts_maps, summary_sva=sess.run([loss, heat_map, loss_summary], feed_dict=feed_dict)
                valid_losses.append(valid_loss)
            # write result figure to the imgs/
            utils.write_result(batch_xs_valid, pts_maps, itr, imgs_dir)
            # save validation log
            validation_writer.add_summary(summary_sva, itr)
            # save the ckpt if reachings better loss
            calc_v_loss = np.mean(valid_losses)

            if calc_v_loss < max_validloss:
                saver.save(sess, logs_dir + "model.ckpt", itr)
                print("[V*] Step: %d, loss:%g" % (itr, calc_v_loss))
                max_validloss = calc_v_loss
                stop_count = 0
            else:
                print("[V] Step: %d, loss:%g" % (itr, calc_v_loss))
                stop_count = stop_count + 1
                if stop_count > (conf.stop_tor + 1):
                    break;
        itr = itr +1    
else:
    print("Testing")
    testing_batch = conf.batch_size
    infered_pts = []
    start_time = time.time()
    for t in tqdm.trange(int(db_helen['testset']['img'].shape[0]/testing_batch)+1):
        t_batch_x = db_helen['testset']['img'][(t*testing_batch):((t+1)*testing_batch)]
        t_batch_y = db_helen['testset']['pts'][(t*testing_batch):((t+1)*testing_batch)]
        if (t == (int(db_helen['testset']['img'].shape[0]/testing_batch)+1)):
            t_batch_x = db_helen['testset']['img'][(t*testing_batch):]
            t_batch_y = db_helen['testset']['pts'][(t*testing_batch):]

        batch_map=sess.run(heat_map, feed_dict={image: t_batch_x/255, keep_probability: 1.0,train_phase:conf.training})

        if (t == 0):
            infered_pts = utils.map2pts(batch_map)
        else:
            infered_pts = np.concatenate((infered_pts, utils.map2pts(batch_map)), axis=0)
            
    used_time = time.time()-start_time
    print('Avg. inference time: %.4f' % (used_time/db_helen['testset']['img'].shape[0]))
    for idx, content in enumerate(zip(db_helen['testset']['img'],infered_pts)):
        img = content[0].copy()
        for kp_idx, keypoint in enumerate(content[1]):
            cv2.circle(img,(keypoint[0],keypoint[1]), 2, (0,255,0), -1)
            
        cv2.imwrite(pred_dir + str(idx)+ '.png', img) 
        
    with open(model_dir + 'pts_'+ str(conf.testing)+ '.pickle', 'wb') as handle:
        pickle.dump(infered_pts, handle)
        
    norm_error_image, norm_error_image_eye = utils.eval_norm_error_image(infered_pts, db_helen['testset']['pts'])
    pandas.DataFrame({'loss':norm_error_image,'loss_eye':norm_error_image_eye}).to_csv(model_dir + 'norm_error_image_' + str(conf.testing)+ '.csv')

sess.close()

