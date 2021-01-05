# coding:utf-8
# 颜色增强
from PIL import ImageEnhance
from PIL import Image
import cv2
import numpy as np

class Color_enhancement():
    def __init__(self):
        pass

    def run(self, img, random_enhance=True, color=1.0, brightness=1.0, contrast=1.0, sharpness=1.0):
        '''
        img:cv2图片
        random_enhance:随机进行增强
        color:色度增强
        brightness:亮度
        contrast:对比度增强
        '''
        # cv2转pil
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  

        if random_enhance:
            color = np.random.randint(8, 13) / 10.  # 随机因子          # 可以用
            brightness = np.random.randint(5, 16) / 10.  # 随机因子   # 这个不用
            contrast = np.random.randint(8, 16) / 10.  # 随机因子     # 这个可以
            sharpness = np.random.randint(0, 16) / 10.  # 随机因子
            # print(color)

        # 色度增强
        if not color == 1.0:
            image = self.color_img(image, color)
        
        # 亮度增强
        if not brightness == 1.0:
            image = self.bright_img(image, brightness)

        # 对比度增强
        if not contrast == 1.0:
            image = self.contrast_img(image, contrast)
        
        # 锐度增强
        if not sharpness == 1.0:
            image = self.sharp_img(image, sharpness)

        # pil转cv2
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  
        return img

    def color_img(self, image, color):
        '''
        image:pil的img
        color:色度增强因子
        '''
        enh_col = ImageEnhance.Color(image)
        image_colored = enh_col.enhance(color)
        return image_colored

    def bright_img(self, image, brightness):
        '''
        image:pil的img
        brightnes:亮度增强因子
        '''
        #亮度增强,增强因子为0.0将产生黑色图像；为1.0将保持原始图像。
        enh_bri = ImageEnhance.Brightness(image)
        image_brightened = enh_bri.enhance(brightness)
        return image_brightened

    # 对比度增强
    def contrast_img(self, image, contrast):
        '''
        image:pil的img
        contrast:对比度增强因子
        '''
        enh_con = ImageEnhance.Contrast(image)
        image = enh_con.enhance(contrast)
        return image

    # 锐度增强
    def sharp_img(self, image, sharpness):
        '''
        image:pil的img
        sharpness:锐度增强因子
        '''
        enh_sha = ImageEnhance.Sharpness(image)
        image = enh_sha.enhance(sharpness)
        return image


if __name__ == "__main__":
    img = cv2.imread("./data/test.jpg")
    enhance = Color_enhancement()
    for i in range(100):
        image = enhance.run(img)
        cv2.imshow("img", image)
        cv2.waitKey(0)