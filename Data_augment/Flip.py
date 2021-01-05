# coding:utf-8
# 翻转图片

import cv2

class Flip():
    def __init__(self):
        pass

    def run(self, img, boxes, vertical=True):
        '''
        img:cv2的图片
        vertical:是否进行竖直翻转
        boxes:整数[[xmin, ymin, xmax, ymax]]
        '''
        height, width, _ = img.shape
        img = self.run_img(img, vertical=vertical)
        boxes = self.run_boxes(boxes, width, height, vertical=vertical)
        return img, boxes
    
    def run_img(self, img, vertical=True):
        '''
        img:cv2的图片
        vertical:是否进行竖直翻转
        '''
        if vertical:
            img = cv2.flip(img, 1)
        return img

    def run_boxes(self, boxes, width, height, vertical=True):
        '''
        boxes:整数[[xmin, ymin, xmax, ymax]]
        vertical:是否进行竖直翻转
        width:图片的宽
        height:图片的高
        '''
        new_boxes = []
        for box in boxes:
            new_box = self.__run_boxes(box, width, height, vertival=vertical)
            new_boxes.append(new_box)
        return new_boxes


    def __run_boxes(self, box, width, height, vertival=True):
        '''
        box:整数[xmin, ymin, xmax, ymax]
        vertical:是否进行竖直翻转
        '''
        if vertival:
            xmin = width - box[2]
            ymin = box[1]
            xmax = width - box[0]
            ymax = box[3]
            return [xmin, ymin, xmax, ymax]
        return box