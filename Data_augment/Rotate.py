# coding:utf-8
# 旋转图片及对应的标签
import cv2
import numpy as np
import math

class Rotate():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pass

    def run(self, img, boxes, angle):
        '''
        img:cv2的图片
        boxes:图片的box, [[xmin, ymin, xmax, ymax]]
        angle:旋转的角度
        '''
        if img is not None:
            img = self.run_img(img, angle)
        if boxes is not None:
            boxes = self.run_boxes(boxes, angle)
        return img, boxes

    def run_img(self, img, angle, scale=1.0):
        '''
        img:cv2图片
        angle:旋转角度
        scale:扩大的倍数
        '''
        if angle is None or angle == 0:
            return img
        w = img.shape[1]
        h = img.shape[0]
        # convet angle into rad
        rangle = np.deg2rad(angle)  # angle in radians
        # calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # map
        return cv2.warpAffine(
            img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
            flags=cv2.INTER_LANCZOS4)

    def run_boxes(self, boxes, angle, scale=1.0):
        '''
        boxes:图片的box, [[xmin, ymin, xmax, ymax]]
        angle:旋转的角度
        '''
        if boxes is None:
            return []

        result = []
        for box in boxes:
            new_box = self.__run_box(box, angle, scale)
            result.append(new_box)
        return result

    def __run_box(self, box, angle, scale=1.0):
        '''
        box:[xmin, xmax, ymin, ymax]
        angle:旋转角度
        scale:缩放尺度
        '''
        w = self.width
        h = self.height
        xmin = box[0]
        xmax = box[1]
        ymin = box[2]
        ymax = box[3]
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        # get width and heigh of changed image
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # rot_mat: the final rot matrix
        # get the four center of edges in the initial martix，and convert the coord
        point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
        # concat np.array
        concat = np.vstack((point1, point2, point3, point4))
        # change type
        concat = concat.astype(np.int32)
        rx, ry, rw, rh = cv2.boundingRect(concat)
        return np.asarray([rx, ry, rw, rh])