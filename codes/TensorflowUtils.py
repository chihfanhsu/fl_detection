__author__ = 'silver'

import cv2
import numpy as np
import random
import imgaug as ia
from imgaug import augmenters as iaa
import tensorflow as tf

IMAGE_SIZE=224

def pts2map(ys):
    #print(ys.shape)
    maps = np.zeros(shape=(ys.shape[0],IMAGE_SIZE,IMAGE_SIZE,1));#print(maps.shape)
    for i in range(ys.shape[0]):
        for p in range(ys.shape[1]):
            if(((ys[i,p,0]) < IMAGE_SIZE) & ((ys[i,p,1]) < IMAGE_SIZE) & (ys[i,p,0] > -1) & (ys[i,p,1] > -1)):
                maps[i, ys[i,p,1], ys[i,p,0], 0] = p+1 # shift label from 0:68 to 1:69, 0-> background, 1:69, landmarks
    return maps

def pts2map_68(ys):
    #print(ys.shape)
    maps = np.zeros(shape=(ys.shape[0],IMAGE_SIZE,IMAGE_SIZE,68));#print(maps.shape)
    for i in range(ys.shape[0]):
        for p in range(ys.shape[1]):
            if(((ys[i,p,0]) < IMAGE_SIZE) & ((ys[i,p,1]) < IMAGE_SIZE) & (ys[i,p,0] > -1) & (ys[i,p,1] > -1)):
                maps[i, ys[i,p,1], ys[i,p,0], p] = 1 
    return maps

def map2pts(pts_maps, gau=False):
    b_idxs = []
    for b in range(pts_maps.shape[0]): 
        idxs = []
        for p in range(pts_maps.shape[3]): 
            idx = np.unravel_index(np.argmax(pts_maps[b,...,p], axis=None), pts_maps[b,...,p].shape)
            idxs.append(idx)
        b_idxs.append(np.asarray(idxs))
    b_idxs = np.asarray(b_idxs);#print(b_idxs.shape)
    if gau == True:
        b_idxs = b_idxs
    else:
        b_idxs = b_idxs[:,1:,:]
    # change index
    ret = np.concatenate((b_idxs[...,1][...,np.newaxis],b_idxs[...,0][...,np.newaxis]),axis=2)
    return ret

# data iterator
def get_batch(X, Y, batch_size = 32):
    # print ('shuffle training dataset')
    idx = np.arange(len(X))    
    while True:
        np.random.shuffle(idx)
        tb = int(len(X)/batch_size)
        #print('total batches %d' % tb)
        for b_idx in range(tb):
            tar_idx = idx[(b_idx*batch_size):((b_idx+1)*batch_size)]
            t_batch_x = X[tar_idx]
            t_batch_y = Y[tar_idx]
            # print(b_idx, t_batch_x.shape, t_batch_y.shape)
            yield t_batch_x, t_batch_y
            
def data_augmentation(images, pts, rot=(-30, 30), s=(0.6, 1.0)):
    keypoints_on_images = []
    for idx_img in range(images.shape[0]):
        image = images[idx_img]
        height, width = image.shape[0:2]
        keypoints = []
        for p in range(pts.shape[1]):
            keypoints.append(ia.Keypoint(x=pts[idx_img,p,0], y=pts[idx_img,p,1]))
        keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=image.shape))

    seq = iaa.Sequential([iaa.Affine(rotate=rot,scale=s)])
    seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start

    # augment keypoints and images
    images_aug = seq_det.augment_images(images)
    keypoints_aug = seq_det.augment_keypoints(keypoints_on_images)
    
    pts_aug=[]
    for img_idx, keypoints_after in enumerate(keypoints_aug):
        img_pts_aug=[]
        for kp_idx, keypoint in enumerate(keypoints_after.keypoints):
            img_pts_aug.append([round(keypoint.x), round(keypoint.y)])
        pts_aug.append(np.asarray(img_pts_aug))

    pts_aug = np.asarray(pts_aug).astype(np.int32)
    
#     print('images_aug', images_aug.shape)
#     print('pts_aug', pts_aug.shape)
    return images_aug, pts_aug

# write image to 
def write_result(batch_xs_valid, pts_maps, iter_num, imgs_dir):
    b = random.randint(0, batch_xs_valid.shape[0]-1)
    img = batch_xs_valid[b].copy()
    pts = map2pts(pts_maps)[b] #print(pts)
#     print(img.shape)
#     print(pts.shape)
    for p in range(pts.shape[0]):
        #print("p",p, pts[p+1,0],pts[p+1,1])
        cv2.circle(img,(pts[p,0],pts[p,1]), 2, (255,0,0), -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(imgs_dir+'infer_'+str(iter_num)+'.png', img) 

def eval_norm_error_image(infer, gt):
    # loss of all landmarks
    l2d = np.sum(np.sqrt(np.sum(np.square(infer-gt),axis=2)), axis=1)
    l2d_eye = np.sum(np.sqrt(np.sum(np.square(infer[:,36:48,:]-gt[:,36:48,:]),axis=2)), axis=1)
    # distance of eye corners
    cd = np.sqrt(np.sum(np.square(gt[:,45,:]-gt[:,36,:]),axis=1))
#     cd = np.sqrt(np.sum(np.square(((gt[:,42,:]+gt[:,45,:])/2)-((gt[:,36,:]+gt[:,39,:])/2)),axis=1))
    norm_error_image = l2d/cd/68
    norm_error_image_eye = l2d_eye/cd/12
    return norm_error_image, norm_error_image_eye

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")