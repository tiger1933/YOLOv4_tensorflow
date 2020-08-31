# coding:utf-8
# test yolov4.weights

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
import os
from os import path
import time
from src.Feature_parse_tf import get_predict_result

def read_img(img_name, width, height):
    img_ori = tools.read_img(img_name)
    if img_ori is None:
        return None, None
    img = cv2.resize(img_ori, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)
    img = img/255.0
    # [416, 416, 3] => [1, 416, 416, 3]
    img = np.expand_dims(img, 0)
    return img, img_ori

def save_img(img, name):
    '''
    img: mat
    '''
    save_dir = "coco_save"
    if not path.isdir(save_dir):
        os.mkdir(save_dir)
    img_name = path.join(save_dir, name)
    cv2.imwrite(img_name, img)
    return 

def main():
    class_num = 80
    width = 608
    height = 608
    score_thresh = 0.5
    iou_thresh = 0.213
    max_box = 50
    anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
    anchors =  np.asarray(anchors).astype(np.float32).reshape([-1, 3, 2])
    model_dir = "./yolo_weights"
    name_file = "./data/coco.names"
    val_dir = "./coco_test_img"
    
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
        ckpt = tf.compat.v1.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("restore model from ", ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("can not find ckpt model")
            assert(0)

        # id to names
        word_dict = tools.get_word_dict(name_file)
        # color of corresponding names
        color_table = tools.get_color_table(class_num)
        
        for name in os.listdir(val_dir):
            img_name = path.join(val_dir, name)
            if not path.isfile(img_name):
                print("'%s' is not a file" %img_name)
                continue

            start = time.perf_counter()

            img, img_ori = read_img(img_name, width, height)
            if img is None:
                continue
            boxes, score, label = sess.run([pre_boxes, pre_score, pre_label], feed_dict={inputs:img})
            
            end = time.perf_counter()
            print("%s\t, time:%f s" %(img_name, end-start))

            img_ori = tools.draw_img(img_ori, boxes, score, label, word_dict, color_table)

            cv2.imshow('img', img_ori)
            cv2.waitKey(0)

            save_img(img_ori, name)

if __name__ == "__main__":
    main()
