# coding:utf-8
# 基本操作

import os
from os import path
import time
import cv2
import random
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

# 将 ls 文件重新写入 file_name 
def rewrite_file(file_name, ls_line):
    '''
    将 ls_line 中的内容写入 file_name
    '''
    with open(file_name, 'w') as f:
        for line in ls_line:
            f.write(str(line) + '\n')
    return

# 解析 voc xml 文件
def parse_voc_xml(file_name, names_dict):
    '''
    解析voc数据集的 xml 文件,每一个列表表示一个图片中的全部标签
    return [ [id1, x1, y1, w1, h1], [id2, x2, y2, w2, h2], ... ]
    '''
    # print(file_name)
    # print(names_dict)
    result = []
    if not os.path.isfile(file_name):
        return None
    doc = parse(file_name)
    root = doc.documentElement
    size = root.getElementsByTagName('size')[0]
    width = int(size.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(size.getElementsByTagName('height')[0].childNodes[0].data)

    objs = root.getElementsByTagName('object')
    for obj in objs:
        name = obj.getElementsByTagName('name')[0].childNodes[0].data
        name_id = names_dict[name]

        bndbox = obj.getElementsByTagName('bndbox')[0]
        xmin = int(float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data))
        ymin = int(float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data))
        xmax = int(float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data))
        ymax = int(float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data))

        x = (xmax + xmin) / 2.0 / width
        w = (xmax - xmin) / width
        y = (ymax + ymin) / 2.0 / height
        h = (ymax - ymin) / height

        result.append([name_id, x, y, w, h])
    return result

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

# 图片画框
def draw_img(img, boxes, score, label, word_dict, color_table,):
    '''
    img : cv2.img [416, 416, 3]
    boxes:[V, 4], x_min, y_min, x_max, y_max
    score:[V], 对应 box 的分数
    label:[V], 对应 box 的标签
    word_dict:id=>name 的变换字典
    window_name:显示的窗口名
    wait_key:暂停时间 ms
    return:画了框的图片
    '''
    w = img.shape[1]
    h = img.shape[0]
    # 字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        boxes[i][0] = constrait(boxes[i][0], 0, 1)
        boxes[i][1] = constrait(boxes[i][1], 0, 1)
        boxes[i][2] = constrait(boxes[i][2], 0, 1)
        boxes[i][3] = constrait(boxes[i][3], 0, 1)
        x_min, x_max = int(boxes[i][0] * w), int(boxes[i][2] * w)
        y_min, y_max = int(boxes[i][1] * h), int(boxes[i][3] * h)
        # 画框
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color_table[label[i]])
        # 写字
        text_name = "{}".format(word_dict[label[i]])
        cv2.putText(img, text_name, (x_min, y_min + 25), font, 1, color_table[label[i]])
        text_score = "{:2d}%".format(int(score[i] * 100))
        cv2.putText(img, text_score, (x_min, y_min), font, 1, color_table[label[i]])
    return img


'''
######################## 其他操作 ####################
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

# 得到随机的色表
def get_color_table(class_num):
    '''
    返回一个包含随机颜色的 (r, g, b) 格式的 list
    '''
    color_table = []
    for i in range(class_num):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color_table.append((b, g, r))
    return color_table


