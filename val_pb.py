# coding:utf-8
# 结果测试

# 解决cudnn 初始化失败的东西: 使用GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import numpy as np
import os
import config
from utils import tools
from utils import data_augment
import cv2

from tensorflow.python.platform import gfile

# 读取图片
def read_img(img_name, width, height):
    '''
    读取一张图片并转化为网络输入格式
    return:网络输入图片, 原始 BGR 图片
    '''
    img_ori = tools.read_img(img_name)
    if img_ori is None:
        return None, None
    if config.keep_img_shape:
        img, nw, nh = data_augment.keep_image_shape_resize(img_ori, size=[width, height])
    else:
        img = cv2.resize(img_ori, (width, height))
        nw, nh = None, None

    show_img = img
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.0
    # [416, 416, 3] => [1, 416, 416, 3]
    img = np.expand_dims(img, 0)
    return img, nw, nh, img_ori, show_img

sess = tf.Session()

# your pb_model path
pb_dir = "./yolo_weights/model.pb"
# your class_num
class_num = 80
# picture folder
test_imgs_folder = "./coco_test_img"
# your names file
names_file = "./data/coco.names"

with gfile.FastGFile(pb_dir, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name="") # 导入计算图
sess.run(tf.global_variables_initializer())

# inputs
inputs = sess.graph.get_tensor_by_name('Placeholder:0')
# output
# 'concat_9', 'concat_10', 'concat_11'
pre_boxes = sess.graph.get_tensor_by_name('concat_9:0')
pre_score = sess.graph.get_tensor_by_name('concat_10:0')  # 
pre_label = sess.graph.get_tensor_by_name('concat_11:0')  # 

width, height = 608, 608

# word_dict = tools.get_word_dict(config.voc_names)     # for VOC
word_dict = tools.get_word_dict(names_file)        # for COCO
color_table = tools.get_color_table(class_num)

for name in os.listdir(test_imgs_folder):
    img_name = os.path.join(test_imgs_folder, name)
    if not os.path.isfile(img_name):
        continue
    img, nw, nh, img_ori, show_img = read_img(img_name, width, height)
    if img is None:
        print("message:'"+str(img)+"'图片读取错误")
    boxes, score, label = sess.run([pre_boxes, pre_score, pre_label], feed_dict={inputs:img})
    
    if config.keep_img_shape:
        # 纠正坐标
        dw = (width - nw)/2
        dh = (height - nh)/2
        for i in range(len(boxes)):
            boxes[i][0] = (boxes[i][0] * width - dw)/nw
            boxes[i][1] = (boxes[i][1] * height - dh)/nh
            boxes[i][2] = (boxes[i][2] * width - dw)/nw
            boxes[i][3] = (boxes[i][3] * height - dh)/nh
    
    img_ori = tools.draw_img(img_ori, boxes, score, label, word_dict, color_table)

    cv2.imshow('img', img_ori)
    cv2.waitKey(0)
