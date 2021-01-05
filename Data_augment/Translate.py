# coding:utf-8
# 图像平移
import cv2
import numpy as np

class Translate():
    def __init__(self, max_width, max_height, pad_thresh=10):
        '''
        max_width:图片的宽
        max_height:图片高
        pad_thresh:平移后至少与边框保持多少的像素,不然就不认为是box
        '''
        self.max_height = max_height
        self.max_width = max_width
        self.pad_thresh = pad_thresh
        pass

    def run(self, img, boxes, x_shift, y_shift, ids=None, max_width=None, max_height=None, back_color=None):
        '''
        img:CV2图片
        boxes:全部box整数坐标[[xmin, ymin, xmax, ymax]]
        x_shift:x坐标平移多少,整数
        y_shift:y坐标平移多少,整数
        ids:每个box对应的id
        back_color:平移过后的背景的值,默认随机值
        max_width:图片最大宽
        max_height:图片最大高
        '''
        if max_width is not None:
            self.max_width = max_width
        if max_height is not None:
            self.max_height = max_height

        trans_img = self.run_img(img, x_shift, y_shift, back_color)
        if ids is None:
            boxes = self.run_boxes(boxes, x_shift, y_shift, max_width=max_width, max_height=max_height)
            return trans_img, boxes
        else:
            boxes, ids = self.run_boxes(boxes, x_shift, y_shift, ids=ids, max_width=max_width, max_height=max_height)
            return trans_img, boxes, ids
    
    def run_img(self, img, x_shift, y_shift, back_color=None):
        '''
        img:CV2图片
        x_shift:x坐标平移多少,整数
        y_shift:y坐标平移多少,整数
        back_color:平移过后的背景的值,默认随机值
        '''
        (height, width) = img.shape[:2]
        # 平移矩阵(浮点数类型)  x_shift +右移 -左移  y_shift -上移 +下移
        matrix = np.float32([[1,0,x_shift],[0,1,y_shift]])
        # 平移图像
        trans_img = cv2.warpAffine(img, matrix, (width, height))
        return trans_img

    def run_boxes(self, boxes, x_shift, y_shift, ids=None, max_width=None, max_height=None):
        '''
        boxes:全部box整数坐标[[xmin, ymin, xmax, ymax]]
        x_shift:x坐标平移多少,整数
        y_shift:y坐标平移多少,整数
        max_width:图片最大宽
        max_height:图片最大高
        '''
        if boxes is None:
            return []
        if max_width is not None:
            self.max_width = max_width
        if max_height is not None:
            self.max_height = max_height
        result = []
        new_ids = []
        for i in range(len(boxes)):
            box = boxes[i]
            curr_id = 0
            if ids is not None:
                curr_id = ids[i]
            trans_box = self.__run_box(box, x_shift, y_shift)
            if trans_box is not None:
                result.append(trans_box)
                new_ids.append(curr_id)
        if ids is not None:
            return result, new_ids
        return result
    
    def __run_box(self, box, x_shift, y_shift):
        '''
        box:box整数坐标[xmin, ymin, xmax, ymax]
        x_shift:x坐标平移多少,整数
        y_shift:y坐标平移多少,整数
        '''
        if box is None:
            return None
        xmin = box[0] + x_shift
        ymin = box[1] + y_shift
        xmax = box[2] + x_shift
        ymax = box[3] + y_shift
        max_width = self.max_width - self.pad_thresh
        max_height =  self.max_height - self.pad_thresh
        if xmin >= max_width:
            return None
        if ymin >= max_height:
            return None
        if xmax <= self.pad_thresh:
            return None
        if ymax <= self.pad_thresh:
            return None
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, self.max_width)
        ymax = min(ymax, self.max_height)
        return [xmin, ymin, xmax, ymax]
    
    # 裁剪指定大小的图片出来
    def __cut_img(self, img, size):
        '''
        TODO:
        '''
        height, width, _ = img.shape
        pad_w = (width - size) // 2
        pad_h = (height - size) // 2
        # new_img = np.zeros(shape=(self.size, self.size, 3), dtype=np.float32)
        new_img = img[pad_h : pad_h+size, pad_w : pad_w+size, : ]
        return new_img
    
if __name__ == "__main__":
    img = cv2.imread("./data/test.jpg")
    cv2.imshow("img", Translate(img.shape[1], img.shape[0]).run_img(img, 20, 50))
    cv2.waitKey(0)