# coding:utf-8
# voc 数据加载
import numpy as np
from src import Log
from utils import tools
from utils import data_augment
import random
import cv2
import os
from os import path

class Data():
    def __init__(self, voc_root_dir, voc_dir_ls, voc_names, class_num, batch_size, anchors, agument, width=608, height=608, data_debug=False):
        self.data_dirs = [path.join(path.join(voc_root_dir, voc_dir), "JPEGImages") for voc_dir in voc_dir_ls]  # 数据文件路径
        self.class_num = class_num  # 分类数
        self.batch_size = batch_size
        self.anchors = np.asarray(anchors).astype(np.float32).reshape([-1, 2]) / [width, height]     #[9,2]
        print("anchors:\n", self.anchors)

        self.imgs_path = []
        self.labels_path = []

        self.num_batch = 0      # 多少个 batch 了
        self.num_imgs = 0       # 一共多少张图片

        self.data_debug = data_debug

        self.width = width
        self.height = height
        self.agument = agument  # data agument strategy

        self.smooth_delta = 0.01 # label smooth delta

        self.names_dict = tools.word2id(voc_names)    # 名字到 id 的字典

        # 初始化各项参数
        self.__init_args()
    
    # 初始化各项参数
    def __init_args(self):
        print("data agument strategy : ", self.agument)
        # 初始化数据增强策略的参数
        self.multi_scale_img = self.agument[0] # 多尺度缩放图片
        self.keep_img_shape = self.agument[1]   # keep image's shape when we reshape the image
        self.flip_img = self.agument[2]    # flip image
        self.gray_img = self.agument[3]        # gray image
        self.label_smooth = self.agument[4]    # label smooth strategy
        self.erase_img = self.agument[5]        # random erase image
        self.invert_img = self.agument[6]                  # invert image pixel
        self.rotate_img = self.agument[7]           # random rotate image

        Log.add_log("message:开始初始化路径")

        # init imgs path
        for voc_dir in self.data_dirs:
            for img_name in os.listdir(voc_dir):
                img_path = path.join(voc_dir, img_name)
                label_path = img_path.replace("JPEGImages", "Annotations")
                label_path = label_path.replace(img_name.split('.')[-1], "xml")
                if not path.isfile(img_path):
                    Log.add_log("warning:VOC 图片文件'"+str(img_path)+"'不存在")
                    continue
                if not path.isfile(label_path):
                    Log.add_log("warning:VOC 标签文件'"+str(label_path)+"'不存在")
                    continue
                self.imgs_path.append(img_path)
                self.labels_path.append(label_path)
                self.num_imgs += 1        
        Log.add_log("message:VOC 数据初始化完成,一共有 "+str(self.num_imgs)+" 张图片")
        
        if self.num_imgs <= 0:
            raise ValueError("没有可训练的图片, 程序退出")
        
        return
        
    # 读取图片
    def read_img(self, img_file):
        '''
        读取 img_file, 并 resize
        return:img, RGB & float
        '''
        img = tools.read_img(img_file)
        if img is None:
            return None

        if self.keep_img_shape:
            # keep image shape when we reshape the image
            img, new_w, new_h = data_augment.keep_image_shape_resize(img, size=[self.width, self.height])
        else:
            img = cv2.resize(img, (self.width, self.height))
            new_w, new_h = None, None

        if self.flip_img:
            # flip image
            img = data_augment.flip_img(img)
        
        if self.gray_img and (np.random.random() < 0.2):
            # probility of gray image is 0.2
            img = data_augment.gray_img(img)
        
        if self.erase_img and (np.random.random() < 0.3):
            # probility of random erase image is 0.3
            img = data_augment.erase_img(img, size_area=[20, 100])
        
        if self.invert_img and (np.random.random() < 0.1):
            # probility of invert image is 0.1
            img = data_augment.invert_img(img)

        if self.rotate_img:
            # rotation image
            img = data_augment.random_rotate_img(img)

        test_img = img

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img/255.0
        return img, new_w, new_h, test_img
    
    # 读取标签
    def read_label(self, label_file, names_dict, anchors, new_w, new_h):
        '''
        读取 label_file, 并生成 label_y1, label_y2, label_y3
        new_w:缩放以后的图片宽,真值
        new_h:缩放以后的图片高,真值
        return:label_y1, label_y2, label_y3
        '''
        contents = tools.parse_voc_xml(label_file, names_dict)  
        if not contents:
            return None, None, None

        # flip the label
        if self.flip_img:
            for i in range(len(contents)):
                contents[i][1] = 1.0 - contents[i][1]

        if self.keep_img_shape:
            x_pad = (self.width - new_w) // 2
            y_pad = (self.height - new_h) // 2

        label_y1 = np.zeros((self.height // 32, self.width // 32, 3, 5 + self.class_num), np.float32)
        label_y2 = np.zeros((self.height // 16, self.width // 16, 3, 5 + self.class_num), np.float32)
        label_y3 = np.zeros((self.height // 8, self.width // 8, 3, 5 + self.class_num), np.float32)

        y_true = [label_y3, label_y2, label_y1]
        ratio = {0:8, 1:16, 2:32}

        test_result = []

        for label in contents:
            label_id = int(label[0])
            box = np.asarray(label[1: 5]).astype(np.float32)   # label中保存的就是 x,y,w,h
            if self.keep_img_shape:
                # 加入填充的黑边宽高，重新修正坐标
                box[0:2] = (box[0:2] * [new_w, new_h ] + [x_pad, y_pad]) / [self.width, self.height]
                box[2:4] = (box[2:4] * [new_w, new_h]) / [self.width, self.height]  
            
            test_result.append([box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2])
            
            best_giou = 0
            best_index = 0
            for i in range(len(anchors)):
                min_wh = np.minimum(box[2:4], anchors[i])
                max_wh = np.maximum(box[2:4], anchors[i])
                giou = (min_wh[0] * min_wh[1]) / (max_wh[0] * max_wh[1])
                if giou > best_giou:
                    best_giou = giou
                    best_index = i
            
            # 012->0, 345->1, 678->2
            x = int(np.floor(box[0] * self.width / ratio[best_index // 3]))
            y = int(np.floor(box[1] * self.height / ratio[best_index // 3]))
            k = best_index % 3

            y_true[best_index // 3][y, x, k, 0:4] = box
            # label smooth
            label_value = 1.0  if not self.label_smooth else ((1-self.smooth_delta) + self.smooth_delta * 1 / self.class_num)
            y_true[best_index // 3][y, x, k, 4:5] = label_value
            y_true[best_index // 3][y, x, k, 5:-1] = 0.0 if not self.label_smooth else self.smooth_delta / self.class_num
            y_true[best_index // 3][y, x, k, 5 + label_id] = label_value
        
        return label_y1, label_y2, label_y3, test_result


    # 加载 batch_size 的数据
    def __get_data(self):
        '''
        加载 batch_size 的标签和数据
        return:imgs, label_y1, label_y2, label_y3
        '''
        # 十个 batch 随机一次 size 
        if self.multi_scale_img and (self.num_batch % 10 == 0):
            random_size = random.randint(13, 23) * 32
            self.width = self.height = random_size
        
        imgs = []
        labels_y1, labels_y2, labels_y3 = [], [], []

        count = 0
        while count < self.batch_size:
            curr_index = random.randint(0, self.num_imgs - 1)
            img_name = self.imgs_path[curr_index]
            label_name = self.labels_path[curr_index]

            # probility of  flip image is 0.5
            if self.agument[2]  and (np.random.random() < 0.5):
                self.flip_img = True
                # print("flip")
            else:
                self.flip_img = False

            img, new_w, new_h, test_img = self.read_img(img_name)
            label_y1, label_y2, label_y3, test_result = self.read_label(label_name, self.names_dict, self.anchors, new_w, new_h)
            
            # 显示最后的结果
            if self.data_debug:
                test_img = tools.draw_img(test_img, test_result, None, None, None, None)
                cv2.imshow("letterbox_img", test_img)
                cv2.waitKey(0)
            
            if img is None:
                Log.add_log("VOC 文件'" + img_name + "'读取异常")
                continue
            if label_y1 is None:
                Log.add_log("VOC 文件'" + label_name + "'读取异常")
                continue
            imgs.append(img)
            labels_y1.append(label_y1)
            labels_y2.append(label_y2)
            labels_y3.append(label_y3)

            count += 1

        self.num_batch += 1
        imgs = np.asarray(imgs)
        labels_y1 = np.asarray(labels_y1)
        labels_y2 = np.asarray(labels_y2)
        labels_y3 = np.asarray(labels_y3)
        
        return imgs, labels_y1, labels_y2, labels_y3

    # 迭代器
    def __next__(self):
        '''
        迭代获得一个 batch 的数据
        '''
        return self.__get_data()

    


