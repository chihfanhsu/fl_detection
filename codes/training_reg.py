
# coding: utf-8

# In[ ]:


import pickle
import os
import sys
import numpy as np
import TensorflowUtils as utils
import tensorflow as tf
import cv2
import pandas
import config
import tqdm
import time
conf, _ = config.get_config()


# In[ ]:


'''setting'''
gpus = [conf.gpu] # Here I set CUDA to only see one GPU
os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(i) for i in gpus])


# In[ ]:


print(conf.tar_model)
if conf.tar_model == 'reg':
    import model_reg as model
elif conf.tar_model == 'dan':
    import model_dan as model
else:
    sys.exit("Sorry, Wrong Model!")
    
model_dir = './'+conf.tar_model+'/'
data_dir = './dataset/'
logs_dir = model_dir+'logs/'
imgs_dir = model_dir+'imgs/'
pred_dir = model_dir+'pred/'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)


# In[ ]:


db_helen = pickle.load(open(data_dir+conf.dataset+".pickle", "rb" ) )
print(db_helen.keys())
print(db_helen['trainset'].keys())
# print the shape of tratining set
print(db_helen['trainset']['pts'].shape)
print(db_helen['trainset']['img'].shape)
# print the shape of testing set
print(db_helen['testset']['pts'].shape)
print(db_helen['testset']['img'].shape)


# In[ ]:


# Define Model Input (x) and Output (y_),  y_ = f(x)
x = tf.placeholder(tf.float32, [None, 224,224,3])
y_ = tf.placeholder(tf.float32, [None,68,2]) # 136
train_phase = tf.placeholder(tf.bool, name='phase_train')
keeprate = tf.placeholder(tf.float32, name="keeprate")
s_mean = tf.placeholder(tf.float32, [None,68,2]) 


# # model 

# In[ ]:


y_out = model.inference(x, s_mean, keeprate, train_phase, False)
y_out_point = tf.reshape(y_out,shape=(-1,68,2))

# Define the Model Loss (4)
avg_losses = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(y_, y_out),-1)),-1))

# Define the Optimizer (5)
train_step = tf.train.AdamOptimizer(conf.lr).minimize(avg_losses)


# In[ ]:


# check parameter
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print('total_parameters', total_parameters)


# In[ ]:


# initialize the model
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

print("Setting up Saver...")
saver = tf.train.Saver(tf.global_variables())


# In[ ]:


sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
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


# In[ ]:


# write image to file
def write_result(batch_xs_valid, batch_pts, iter_num):
    b = random.randint(0, batch_pts.shape[0]-1)
    img = batch_xs_valid[b].copy()
    pts = batch_pts[b] #print(pts)
    for p in range(pts.shape[0]):
        #print("p",p, pts[p+1,0],pts[p+1,1])
        cv2.circle(img,(pts[p,0],pts[p,1]), 2, (255,0,0), -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(imgs_dir + 'infer_'+str(iter_num)+'.png', img)


# In[ ]:


batches = utils.get_batch(db_helen['trainset']['img'], db_helen['trainset']['pts'], batch_size = conf.batch_size)
valid_batches = utils.get_batch(db_helen['testset']['img'], db_helen['testset']['pts'], batch_size = conf.batch_size)


# In[ ]:


with open('avg_shape.pickle', 'rb') as handle:
    avg_shape = pickle.load(handle)

if conf.training == True:
    step = 0
    stop_count = 0
    max_validloss = 99999
    while True:
        batch_xs, batch_ys = next(batches)
        batch_xs_aug, batch_ys_aug =utils.data_augmentation(batch_xs, batch_ys)
        

        sess.run([extra_update_ops,train_step], feed_dict={x: batch_xs_aug/255,
                                                           s_mean:avg_shape,
                                                           y_: batch_ys_aug,
                                                           train_phase: True,
                                                           keeprate:0.5})
        if step % 500 == 0:
            train_loss = sess.run(avg_losses, feed_dict={x: batch_xs_aug/255,
                                                           s_mean:avg_shape,
                                                           y_: batch_ys_aug,
                                                           train_phase: False,
                                                           keeprate:1.0})
            print("[T] Step: %d, loss:%g" % (step, np.mean(train_loss)))
            
        if (step % 1000 == 0):
            # get training accr
            # calculate all batches
            valid_losses = []
            for i in tqdm.trange(int((db_helen['testset']['pts'].shape[0])/conf.batch_size)):
                t_batch_x, t_batch_y = next(valid_batches)
                infered_pts, valid_loss= sess.run([y_out_point, avg_losses], feed_dict={x: t_batch_x/255,
                                                                                         s_mean:avg_shape,
                                                                                         y_: t_batch_y,
                                                                                         train_phase: False,
                                                                                         keeprate:1.0})
                valid_losses.append(valid_loss)
            write_result(t_batch_x, infered_pts, step)
            
            calc_v_loss = np.mean(valid_losses)

            if calc_v_loss < max_validloss:
                saver.save(sess, logs_dir + "model.ckpt", step)
                print("[V*] Step: %d, loss:%g" % (step, calc_v_loss))
                max_validloss = calc_v_loss
                stop_count = 0
            else:
                print("[V] Step: %d, loss:%g" % (step, calc_v_loss))
                stop_count = stop_count + 1
                if stop_count > (conf.stop_tor + 1):
                    break;
        step = step +1
        
else: # evaluate
    testing_batch = conf.batch_size
    inferred_map = []
    start_time = time.time()
    for t in tqdm.trange(int(db_helen['testset']['img'].shape[0]/testing_batch)+1):
        t_batch_x = db_helen['testset']['img'][(t*testing_batch):((t+1)*testing_batch)]
        t_batch_y = db_helen['testset']['pts'][(t*testing_batch):((t+1)*testing_batch)]

        if (t == (int(db_helen['testset']['img'].shape[0]/testing_batch)+1)):
            t_batch_x = db_helen['testset']['img'][(t*testing_batch):]
            t_batch_y = db_helen['testset']['pts'][(t*testing_batch):]
            
        feed_dict={x: t_batch_x/255,
                   s_mean:avg_shape,
                   y_: t_batch_y,
                   train_phase: False,
                   keeprate:1.0}

        infered_pts, acc_valid= sess.run([y_out_point, avg_losses],
                                         feed_dict=feed_dict)

        if (t == 0):
            inferred_map = infered_pts
        else:
            inferred_map = np.concatenate((inferred_map, infered_pts), axis=0)
            
    used_time = time.time()-start_time
    print('Avg. inference time: %.4f' % (used_time/db_helen['testset']['img'].shape[0]))
    inferred_map = np.asarray(inferred_map)

    pts_maps = np.reshape(inferred_map, newshape=(-1,inferred_map.shape[1],inferred_map.shape[2]))
    
    norm_error_image, norm_error_image_eye= utils.eval_norm_error_image(pts_maps, db_helen['testset']['pts'])
    
    with open(model_dir + 'pts_'+ str(conf.testing)+ '.pickle', 'wb') as handle:
        pickle.dump(pts_maps, handle)
        
    pandas.DataFrame({'loss':norm_error_image,'loss_eye':norm_error_image_eye}).to_csv(model_dir + 'norm_error_image_' + str(conf.testing)+ '.csv')
    for idx, content in enumerate(zip(db_helen['testset']['img'],pts_maps)):
        img = content[0].copy()
        for kp_idx, keypoint in enumerate(content[1]):
            cv2.circle(img,(keypoint[0],keypoint[1]), 2, (0,255,0), -1)

        cv2.imwrite(pred_dir + str(idx)+ '.png', img)

        
sess.close()


# In[ ]:




