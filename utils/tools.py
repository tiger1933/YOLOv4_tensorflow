# coding:utf-8
# 基本操作

import os
from os import path
import time
import math
import cv2
import random
import numpy as np
from utils.k_means import get_kmeans
from xml.dom.minidom import parse

'''
##################### 文件操作 #####################
'''
# 读取文件全部内容
def read_file(file_name):
    '''
    读取 file_name 文件全部内容
    return:文件内容list
    '''
    if not path.isfile(file_name):
        return None
    result = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            # 去掉换行符和空格
            line = line.strip('\n').strip()
            if len(line) == 0:
                continue
            result.append(line)
    return result

# 写入文件,是否写入时间
def write_file(file_name, line, write_time=False):
    '''
    file_name:写入文件名
    line:写入文件内容
    write_time:是否在内容前一行写入时间
    '''
    with open(file_name,'a') as f:
        if write_time:
            line = get_curr_data() + '\n' + str(line)
        f.write(str(line) + '\n')
    return None

# 将列表写入文件,是否写入时间
def write_file_ls(file_name, line_ls, write_time=False):
    '''
    file_name:写入文件名
    line_ls:写入文件的列表内容
    write_time:是否在内容前一行写入时间
    '''
    with open(file_name,'a') as f:
        for line in line_ls:
            if write_time:
                line = get_curr_data() + '\n' + str(line)
            f.write(str(line) + '\n')
    return None

# 将 ls 文件重新写入 file_name 
def rewrite_file(file_name, ls_line):
    '''
    将 ls_line 中的内容写入 file_name
    '''
    with open(file_name, 'w') as f:
        for line in ls_line:
            f.write(str(line) + '\n')
    return

# 得到文件夹下的全部子文件夹
def get_sub_dir(root_dir):
    result = []
    for name in os.listdir(root_dir):
        if name[0] == '.':
            continue
        curr_name = path.join(root_dir, name)
        if path.isdir(curr_name):
            result.append(name)
    return result

# 解析 voc xml 文件
def parse_voc_xml(file_name, width=None, height=None, names_dict=None, get_ori=False):
    '''
    解析voc数据集的 xml 文件,每一个列表表示一个图片中的全部标签
    names_dict:名字到id的字典
    if names_dict is not None:
        return [ [id1, x1, y1, w1, h1], [id2, x2, y2, w2, h2], ... ]
    if names_dict is None:
        return [ [x1, y1, w1, h1], [x2, y2, w2, h2], ... ]
    get_ori:不进行浮点化,[[xmin, ymin, xmax, ymax]]
    '''
    # print(file_name)
    # print(names_dict)
    result = []
    if not os.path.isfile(file_name):
        print(file_name+"不是文件")
        return None
    doc = parse(file_name)
    root = doc.documentElement
    # # 是没有 size 属性的
    # size = root.getElementsByTagName('size')[0]
    # width = int(size.getElementsByTagName('width')[0].childNodes[0].data)
    # height = int(size.getElementsByTagName('height')[0].childNodes[0].data)

    objs = root.getElementsByTagName('object')
    for obj in objs:
        bndbox = obj.getElementsByTagName('bndbox')[0]
        xmin = int(float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data))
        ymin = int(float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data))
        xmax = int(float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data))
        ymax = int(float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data))

        if not get_ori:
            assert(width is not None)
            assert(height is not None)
            x = (xmax + xmin) / 2.0 / width
            w = (xmax - xmin) / width
            y = (ymax + ymin) / 2.0 / height
            h = (ymax - ymin) / height
        else:
            x = xmin
            y = ymin 
            w = xmax
            h = ymax

        if names_dict is not None:
            name = obj.getElementsByTagName('name')[0].childNodes[0].data
            name_id = names_dict[name]
            result.append([name_id, x, y, w, h])
        else:
            result.append([x, y, w, h])

    return result

# 判断是不是图片
def is_pic(name):
    if name.split('.')[-1] in ['jpg','png','jpeg']:
        return True
    return False

'''
######################## 时间操作 ####################
'''
# 获得当前日期
def get_curr_data():
    '''
    return : 年-月-日-时-分-秒
    '''
    t = time.gmtime()
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S",t)
    return time_str

'''
######################## 图片操作 ####################
'''
# 读取图片
def read_img(file_name):
    '''
    以 BGR 格式读取图片
    return:BGR图片
    '''
    if not path.exists(file_name):
        return None
    img = cv2.imread(file_name)
    return img

# 读取填充后的图片
def read_img_pad(file_name, scale=32):
    '''
    file_name:图片名
    以 scale 为单位进行填充
    return:BGR填充图片和原始图片BGR图片
    '''
    if not path.exists(file_name):
        return None,None
    img_ori = cv2.imread(file_name)
    ori_h , ori_w , _= img_ori.shape
    height = math.ceil(ori_h/scale) * scale
    width = math.ceil(ori_w/scale) * scale
    img = np.full(shape = [height , width , 3] , fill_value= 0, dtype=np.uint8)
    dh , dw = (height - ori_h)//2 , (width - ori_w)//2
    img[dh:(ori_h+dh) , dw:(ori_w+dw),:] = img_ori
    return img, img_ori

# 图片画框
def draw_img(img, boxes, score=None, label=None, word_dict=None, color_table=None,):
    '''
    img : cv2.img [416, 416, 3]
    boxes:[V, 4],float x_min, y_min, x_max, y_max
    score:[V], 对应 box 的分数
    label:[V], 对应 box 的标签
    word_dict:id=>name 的变换字典
    return:画了框的图片
    '''
    w = img.shape[1]
    h = img.shape[0]
    # 字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        # boxes[i][0] = constrait(boxes[i][0], 0, 1)
        # boxes[i][1] = constrait(boxes[i][1], 0, 1)
        # boxes[i][2] = constrait(boxes[i][2], 0, 1)
        # boxes[i][3] = constrait(boxes[i][3], 0, 1)
        # x_min, x_max = int(boxes[i][0] * w), int(boxes[i][2] * w)
        # y_min, y_max = int(boxes[i][1] * h), int(boxes[i][3] * h)
        xmin = boxes[i][0] * w
        x_min = int(xmin if xmin > 0 else 0)
        ymin = boxes[i][1] * h
        y_min = int(ymin if ymin > 0 else 0)
        if (xmin >= w) or (ymin >= h):
            continue
        xmax = boxes[i][2] * w
        x_max = int(xmax if xmax < w else w)
        ymax = boxes[i][3] * h
        y_max = int(ymax if ymax < h else h)
        if (xmax <= 0) or (ymax <=0):
            continue
        # 画框
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color_table[label[i]] if color_table is not None else [123, 123, 255], 2)
        # 写字
        if word_dict is not None:
            text_name = "{}".format(word_dict[label[i]])
            cv2.putText(img, text_name, (x_min, y_min + 25), font, 1, color_table[label[i]])
        if score is not None:
            text_score = "{:2d}%".format(int(np.ceil(score[i] * 100)))
            cv2.putText(img, text_score, (x_min, y_min), font, 1, color_table[label[i]])
    return img

# 图片画框
def draw_box_float(img, boxes):
    '''
    img : cv2.img [416, 416, 3]
    boxes:[V, 4],float x, y, w, h
    return:画了框的图片
    '''
    width = img.shape[1]
    height = img.shape[0]
    for box in boxes:
        x, y, w, h = box
        x_min, x_max = int((x-w/2) * width), int((x+w/2) * width)
        y_min, y_max = int((y-h/2) * height), int((y+h/2) * height)
        # 画框
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), [123, 123, 255], 2)
    return img

'''
#################### 其他操作 ####################
'''
# 得到 id => name 的转换
def get_word_dict(name_file):
    '''
    得到 id 到 名字的字典
    return:{}
    '''
    word_dict = {}
    contents = read_file(name_file)
    for i in range(len(contents)):
        word_dict[i] = str(contents[i])
    return word_dict

# name => id 的转换
def word2id(names_file):
    '''
    得到 名字 到 id 的转换字典
    return {}
    '''
    id_dict = {}
    contents = read_file(names_file)
    for i in range(len(contents)):
        id_dict[str(contents[i])] = i
    return id_dict

# 限制数字
def constrait(x, start, end):
    '''    
    将 x 限制到 [start, end] 闭区间之间
    return:x    ,start <= x <= end
    '''
    if x < start:
        return start
    elif x > end:
        return end
    else:
        return x

#     判断这个字符串是不是数字
def is_number(s):
    try:
        float(s)
        return True
    except:
        return False
    return False

# 得到随机的色表
def get_color_table(class_num):
    '''
    返回一个包含随机颜色的 (r, g, b) 格式的 list
    '''
    color_table = []
    for i in range(class_num):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(128, 255)
        color_table.append((b, g, r))
    return color_table

# 聚类得到 anchors 
def get_anchors(ls_xml_path, ls_img_path, target_size=None, k=6, min_box = (16, 16)):
    '''
    读取 anchors
    '''
    result = []
    # 读取 width and height
    print("anchors_file不存在,使用聚类生成")
    anno_result = []
    num_dict = {}
    count_no_size = 0
    total_index = len(ls_xml_path)
    for i in range(total_index):
        if i % 100 == 0:
            print("\r当前进度{:03f}%".format((i+1)/total_index*100), end="")
        xml_name =ls_xml_path[i]
        img_name = ls_img_path[i]
        ori_h, ori_w, _ = read_img(img_name).shape
        # [x, y, w, h] float
        boxs = parse_voc_xml(xml_name, ori_w, ori_h, None)
        if boxs is None:
            count_no_size += 1
            continue
        if len(boxs) not in num_dict:
            num_dict[len(boxs)] = 1
        else:
            num_dict[len(boxs)] = num_dict[len(boxs)] + 1

        for i in range(len(boxs)):
            if target_size is not None:
                width = math.ceil(boxs[i][2] * target_size[0])
                height = math.ceil(boxs[i][3] * target_size[1])
            else:
                width = math.ceil(boxs[i][2] * ori_w)
                height = math.ceil(boxs[i][3] * ori_h)
            anno_result.append([width, height])
            
    # # 打印 anno_result 结果
    # print("\nanno_result:")
    # # for i in range(min(len(anno_result), 30)):
    # curr_ls = []
    # for i in range(len(anno_result)):
    #     if i%10 == 0:
    #         print(curr_ls)
    #         curr_ls = []
    #     curr_ls.append(anno_result[i][0])
    #     curr_ls.append(anno_result[i][1])
    #     pass
    # print(curr_ls)

    print("\n正在聚类")
    anno_result = np.asarray(anno_result)
    # k_means
    anchors, ave_iou = get_kmeans(anno_result, k)
    for anchor in anchors:
        result.append(anchor[0])
        result.append(anchor[1])
    print("generate anchors:", result)
    print("miou:", ave_iou)
    # print("目标个数统计:\n", num_dict)
    # print("没有size个数统计:\n", count_no_size)

    return result
