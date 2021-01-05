# coding:utf-8
# 随机擦除图像
import numpy as np

class Erase():
    def __init__(self, erase_num=None, max_erase=10, random_size=True, min_w=1, min_h=1, max_w=60, max_h=60, fit_value=None):
        '''
        erase_num:擦除的方框个数
        max_erase:如果erase_num是None,就随机生成一个小于它数字进行擦除
        random_size:是否随机的进行擦除,否则擦除[max_w,max_h]的区域
        min_w:擦除区域的最小宽度
        min_h:擦除区域的最小高度
        max_w:擦除区域的最大宽度
        max_h:擦除区域的最大高度
        fit_value:擦除区域填充的像素值,默认随机值
        '''
        self.erase_num = erase_num
        self.max_erase = max_erase
        self.random_size = random_size
        self.min_w = min_w
        self.max_w = max_w
        self.min_h = min_h
        self.max_h = max_h
        self.fit_value = fit_value
        pass

    def run(self, img):
        '''
        img:cv2的图片
        '''
        # 随机擦除
        height, width, _ = img.shape
        min_w = self.min_w
        max_w = self.max_w
        min_h = self.min_h
        max_h = self.max_h
        erase_num = np.random.randint(self.max_erase)
        if self.erase_num:
            erase_num = self.erase_num
        for i in range(erase_num):
            erase_w = max_w
            erase_h = max_h
            if self.random_size:
                erase_w = np.random.randint(min_w, max_w)
                erase_h = np.random.randint(min_h, max_h)
            x = np.random.randint(0, width - erase_w)
            y = np.random.randint(0, height - erase_h)
            value = self.fit_value if self.fit_value else np.random.randint(0, 255)
            img[y:y+erase_h, x:x+erase_w, : ] = value
        return img