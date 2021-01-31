# coding:utf-8
# test on voc

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import config
from utils import tools
from src.YOLO import YOLO
import cv2
import numpy as np
from src import Log
import os
from os import path
import time
from src.Feature_parse_tf import get_predict_result

class_num = config.voc_class_num
width = config.width
height = config.height
anchors =  np.asarray(config.voc_anchors).astype(np.float32).reshape([-1, 3, 2]) if config.voc_anchors else None
score_thresh = config.val_score_thresh
iou_thresh = config.val_iou_thresh
max_box = config.max_box
model_path = config.voc_model_path
name_file = config.voc_names
val_dir = config.voc_test_dir
save_img = config.save_img
save_dir = config.voc_save_dir

def read_img(img_name, width, height):
    img_ori = tools.read_img(img_name)
    if img_ori is None:
        return None, None
    ori_h, ori_w, _ = img_ori.shape
    img = cv2.resize(img_ori, (width, height))

    show_img = img
    
    img = img.astype(np.float32)
    img = img/255.0
    # [416, 416, 3] => [1, 416, 416, 3]
    img = np.expand_dims(img, 0)
    return img, ori_w, ori_h, img_ori, show_img

def write_img(img, name):
    '''
    img: mat 
    name: saved name
    '''
    if not path.isdir(save_dir):
        Log.add_log("message: create floder'"+str(save_dir)+"'")
        os.mkdir(save_dir)
    img_name = path.join(save_dir, name)
    cv2.imwrite(img_name, img)
    return 

def main():
    yolo = YOLO()

    inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
    feature_y1, feature_y2, feature_y3 = yolo.forward(inputs, class_num, isTrain=False)
    pre_boxes, pre_score, pre_label = get_predict_result(feature_y1, feature_y2, feature_y3,
                                                                                                anchors[2], anchors[1], anchors[0], 
                                                                                                width, height, class_num, 
                                                                                                score_thresh=score_thresh, 
                                                                                                iou_thresh=iou_thresh,
                                                                                                max_box=max_box)

    init = tf.compat.v1.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        ckpt = tf.compat.v1.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            Log.add_log("message: load ckpt model:'"+str(ckpt.model_checkpoint_path)+"'")
        else:
            Log.add_log("message:can not find  ckpt model")
            # exit(1)
            # assert(0)
        
        # dictionary of name of corresponding id
        word_dict = tools.get_word_dict(name_file)
        # dictionary of per names
        color_table = tools.get_color_table(class_num)
        
        for name in os.listdir(val_dir):
            img_name = path.join(val_dir, name)
            if not path.isfile(img_name):
                print("'%s' is not file" %img_name)
                continue

            img, nw, nh, img_ori, show_img = read_img(img_name, width, height)
            if img is None:
                Log.add_log("message:'"+str(img)+"' is None")
                continue

            start = time.perf_counter()
            
            boxes, score, label = sess.run([pre_boxes, pre_score, pre_label], feed_dict={inputs:img})
            
            end = time.perf_counter()
            print("%s\t, time:%f s" %(img_name, end-start))
           
            img_ori = tools.draw_img(img_ori, boxes, score, label, word_dict, color_table)
            cv2.imshow('img', img_ori)
            cv2.waitKey(0)

            if save_img:
                write_img(img_ori, name)
            pass

if __name__ == "__main__":
    Log.add_log("message: into val.main()")
    main()
