# coding:utf-8
# YOLO实现
import tensorflow as tf
from src import module
import numpy as np
slim = tf.contrib.slim

class YOLO():
    def __init__(self,class_num, anchors, width=608, height=608):
        self.class_num = class_num
        self.anchors = np.asarray(anchors).reshape([-1, 3, 2])
        self.width = width
        self.height = height
        pass

    def forward(self, inputs, batch_norm_decay=0.9, weight_decay=0.0005, isTrain=True, reuse=False):
        # set batch norm params
        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': isTrain,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            # darknet53 特征
            # [N, 19, 19, 512], [N, 38, 38, 256], [N, 76, 76, 128]
            route_1, route_2, route_3 = module.extraction_feature(inputs, batch_norm_params, weight_decay)
            
            with slim.arg_scope([slim.conv2d], 
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                with tf.variable_scope('yolo'):
                    # 计算 y1 特征
                    # [N, 76, 76, 128] => [N, 76, 76, 256]
                    net = module.conv(route_1, 256)
                    # [N, 76, 76, 256] => [N, 76, 76, 255]
                    net = slim.conv2d(net, 3*(4+1+self.class_num), 1,
                                                        stride=1, normalizer_fn=None,
                                                        activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_y3 = net

                    # 计算 y2 特征
                    # [N, 76, 76, 128] => [N, 38, 38, 256]
                    net = module.conv(route_1, 256, stride=2)
                    # [N, 38, 38, 512]
                    net = tf.concat([net, route_2], -1)
                    net = module.yolo_conv_block(net, 512, 2, 1)
                    route_147 = net
                    # [N, 38, 38, 256] => [N, 38, 38, 512]
                    net = module.conv(net, 512)
                    # [N, 38, 38, 512] => [N, 38, 38, 255]
                    net = slim.conv2d(net, 3*(4+1+self.class_num), 1,
                                                        stride=1, normalizer_fn=None,
                                                        activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_y2 = net

                    # 计算 y3 特征
                    # [N, 38, 38, 256] => [N, 19, 19, 512]
                    net = module.conv(route_147, 512, stride=2)
                    net = tf.concat([net, route_3], -1)
                    net = module.yolo_conv_block(net, 1024, 3, 0)
                    # [N, 19, 19, 1024] => [N, 19, 19, 255]
                    net = slim.conv2d(net, 3*(4+1+self.class_num), 1,
                                                        stride=1, normalizer_fn=None,
                                                        activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_y1 = net

        return feature_y1, feature_y2, feature_y3

    # 计算最大的 IOU, GIOU
    def IOU(self, pre_xy, pre_wh, valid_yi_true):
        '''
            pre_xy : [13, 13, 3, 2]
            pre_wh : [13, 13, 3, 2]
            valid_yi_true : [V, 5 + class_num] or [V, 4]
            return:
                iou, giou : [13, 13, 3, V], V表示V个真值
        '''

        # [13, 13, 3, 2] ==> [13, 13, 3, 1, 2]
        pre_xy = tf.expand_dims(pre_xy, -2)
        pre_wh = tf.expand_dims(pre_wh, -2)

        # [V, 2]
        yi_true_xy = valid_yi_true[..., 0:2]
        yi_true_wh = valid_yi_true[..., 2:4]

        # 相交区域左上角坐标 : [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersection_left_top = tf.maximum((pre_xy - pre_wh / 2), (yi_true_xy - yi_true_wh / 2))
        # 相交区域右下角坐标
        intersection_right_bottom = tf.minimum((pre_xy + pre_wh / 2), (yi_true_xy + yi_true_wh / 2))
        # 并集区域左上角坐标 
        combine_left_top = tf.minimum((pre_xy - pre_wh / 2), (yi_true_xy - yi_true_wh / 2))
        # 并集区域右下角坐标
        combine_right_bottom = tf.maximum((pre_xy + pre_wh / 2), (yi_true_xy + yi_true_wh / 2))

        # 相交区域宽高 [13, 13, 3, V, 2] == > [13, 13, 3, V, 2]
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        # 并集区域宽高 : 这里其实不用 tf.max 也可以，因为右下坐标一定大于左上
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)
        
        # 相交区域面积 : [13, 13, 3, V]
        intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]
        # 预测box面积 : [13, 13, 3, 1]
        pre_area = pre_wh[..., 0] * pre_wh[..., 1]
        # 真值 box 面积 : [V]
        true_area = yi_true_wh[..., 0] * yi_true_wh[..., 1]
        # [V] ==> [1, V]
        true_area = tf.expand_dims(true_area, axis=0)
        # iou : [13, 13, 3, V]
        iou = intersection_area / (pre_area + true_area - intersection_area + 1e-10)    # 防止除0

        # 并集区域面积 : [13, 13, 3, V, 1] ==> [13, 13, 3, V] 
        combine_area = combine_wh[..., 0] * combine_wh[..., 1]
        # giou : [13, 13, 3, V]
        giou = (intersection_area+1e-10) / combine_area # 加上一个很小的数字，确保 giou 存在
        
        return iou, giou

    # 计算 CIOU 损失
    def __my_CIOU_loss(self, pre_xy, pre_wh, yi_box):
        '''
        the formula of CIOU_LOSS is refers to http://bbs.cvmart.net/topics/1436
        计算每一个 box 与真值的 ciou 损失
        pre_xy:[batch_size, 13, 13, 3, 2]
        pre_wh:[batch_size, 13, 13, 3, 2]
        yi_box:[batch_size, 13, 13, 3, 4]
        return:[batch_size, 13, 13, 3, 1]
        '''
        # [batch_size, 13, 13, 3, 2]
        yi_true_xy = yi_box[..., 0:2]
        yi_true_wh = yi_box[..., 2:4]

        # 上下左右
        pre_lt = pre_xy - pre_wh/2
        pre_rb = pre_xy + pre_wh/2
        truth_lt = yi_true_xy - yi_true_wh / 2
        truth_rb = yi_true_xy + yi_true_wh / 2

        # 相交区域坐标 : [batch_size, 13, 13, 3,2]
        intersection_left_top = tf.maximum(pre_lt, truth_lt)
        intersection_right_bottom = tf.minimum(pre_rb, truth_rb)
        # 相交区域宽高 : [batch_size, 13, 13, 3, 2]
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        # 相交区域面积 : [batch_size, 13, 13, 3, 1]
        intersection_area = intersection_wh[..., 0:1] * intersection_wh[..., 1:2]
        # 并集区域左上角坐标 
        combine_left_top = tf.minimum(pre_lt, truth_lt)
        # 并集区域右下角坐标
        combine_right_bottom = tf.maximum(pre_rb, truth_rb)
        # 并集区域宽高 : 这里其实不用 tf.max 也可以，因为右下坐标一定大于左上
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)

        # 并集区域对角线 : [batch_size, 13, 13, 3, 1]
        C = tf.square(combine_wh[..., 0:1]) + tf.square(combine_wh[..., 1:2])
        # 中心点对角线:[batch_size, 13, 13, 3, 1]
        D = tf.square(yi_true_xy[..., 0:1] - pre_xy[..., 0:1]) + tf.square(yi_true_xy[..., 1:2] - pre_xy[..., 1:2])

        # box面积 : [batch_size, 13, 13, 3, 1]
        pre_area = pre_wh[..., 0:1] * pre_wh[..., 1:2]
        true_area = yi_true_wh[..., 0:1] * yi_true_wh[..., 1:2]

        # iou : [batch_size, 13, 13, 3, 1]
        iou = intersection_area / (pre_area + true_area - intersection_area)

        pi = 3.14159265358979323846

        # 衡量长宽比一致性的参数:[batch_size, 13, 13, 3, 1]
        v = 4 / (pi * pi) * tf.square( 
                                      tf.math.atan(yi_true_wh[..., 0:1] / yi_true_wh[..., 1:2])
                                       - tf.math.atan(pre_wh[...,0:1] / pre_wh[..., 1:2]) )

        # trade-off 参数
        # alpha
        alpha = v / (1.0 - iou + v)
        # alpha = 1.0
        ciou_loss = 1 - iou + D / C + alpha * v
        return ciou_loss

    # 计算 GIOU 损失
    def __my_GIOU_loss(self, pre_xy, pre_wh, yi_box):
        '''
        the formula of GIOU_LOSS is refers to http://bbs.cvmart.net/topics/1436
        计算每一个 box 与真值的 ciou 损失
        pre_xy:[batch_size, 13, 13, 3, 2]
        pre_wh:[batch_size, 13, 13, 3, 2]
        yi_box:[batch_size, 13, 13, 3, 4]
        return:[batch_size, 13, 13, 3, 1]
        '''
        # [batch_size, 13, 13, 3, 2]
        yi_true_xy = yi_box[..., 0:2]
        yi_true_wh = yi_box[..., 2:4]

        # 上下左右
        pre_lt = pre_xy - pre_wh/2
        pre_rb = pre_xy + pre_wh/2
        truth_lt = yi_true_xy - yi_true_wh / 2
        truth_rb = yi_true_xy + yi_true_wh / 2

        # 相交区域坐标 : [batch_size, 13, 13, 3,2]
        intersection_left_top = tf.maximum(pre_lt, truth_lt)
        intersection_right_bottom = tf.minimum(pre_rb, truth_rb)
        # 相交区域宽高 : [batch_size, 13, 13, 3, 2]
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        # 相交区域面积 : [batch_size, 13, 13, 3, 1]
        intersection_area = intersection_wh[..., 0:1] * intersection_wh[..., 1:2]
        # 并集区域左上角坐标 
        combine_left_top = tf.minimum(pre_lt, truth_lt)
        # 并集区域右下角坐标
        combine_right_bottom = tf.maximum(pre_rb, truth_rb)
        # 并集区域宽高 : 这里其实不用 tf.max 也可以，因为右下坐标一定大于左上
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)

        # 并集区域面积 : [batch_size, 13, 13, 3, 1]
        C = combine_wh[..., 0:1] * combine_wh[..., 1:2]

        # box面积 : [batch_size, 13, 13, 3, 1]
        pre_area = pre_wh[..., 0:1] * pre_wh[..., 1:2]
        true_area = yi_true_wh[..., 0:1] * yi_true_wh[..., 1:2]

        # iou : [batch_size, 13, 13, 3, 1]
        iou = intersection_area / (pre_area + true_area - intersection_area + 1e-10)
        giou = iou - (C - (pre_area + true_area - intersection_area)) / C
        giou_loss = 1.0 - giou
        return giou_loss

    # 计算 CIOU 损失
    def __get_CIOU_loss(self, pre_xy, pre_wh, yi_box):
        '''
        the formula of CIOU_LOSS is refers to http://bbs.cvmart.net/topics/1436
        the implement of this function is refers to yolov4 -> box.c -> dx_box_iou function
        计算每一个 box 与真值的 ciou 损失
        pre_xy:[batch_size, 13, 13, 3, 2]
        pre_wh:[batch_size, 13, 13, 3, 2]
        yi_box:[batch_size, 13, 13, 3, 4]
        return:[batch_size, 13, 13, 3, 4]
        '''
        # [batch_size, 13, 13, 3, 2]
        yi_true_xy = yi_box[..., 0:2]
        yi_true_wh = yi_box[..., 2:4]

        pre_lt = pre_xy - pre_wh/2
        pre_rb = pre_xy + pre_wh/2
        truth_lt = yi_true_xy - yi_true_wh / 2
        truth_rb = yi_true_xy + yi_true_wh / 2

        # 相交区域坐标 : [batch_size, 13, 13, 3,2]
        intersection_left_top = tf.maximum(pre_lt, truth_lt)
        intersection_right_bottom = tf.minimum(pre_rb, truth_rb)
        # 相交区域宽高 : [batch_size, 13, 13, 3, 2]
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        lw = intersection_wh[..., 0:1]
        lh = intersection_wh[..., 1:2]
        # 相交区域面积 : [batch_size, 13, 13, 3, 1]
        intersection_area = intersection_wh[..., 0:1] * intersection_wh[..., 1:2]
        # 并集区域左上角坐标 
        combine_left_top = tf.minimum(pre_lt, truth_lt)
        # 并集区域右下角坐标
        combine_right_bottom = tf.maximum(pre_rb, truth_rb)
        # 并集区域宽高 : 这里其实不用 tf.max 也可以，因为右下坐标一定大于左上
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)
        # giou_Ch = combine_wh[..., 1:2]
        # giou_Cw = combine_wh[..., 0:1]

        # box面积 : [batch_size, 13, 13, 3, 1]
        pre_area = pre_wh[..., 0:1] * pre_wh[..., 1:2]
        true_area = yi_true_wh[..., 0:1] * yi_true_wh[..., 1:2]
        # 并集区域面积 : [batch_size, 13, 13, 3, 1]
        combine_area = combine_wh[..., 0:1] * combine_wh[..., 1:2]

        # iou : [batch_size, 13, 13, 3, 1]
        iou = intersection_area / (pre_area + true_area - intersection_area)
        U = pre_area + true_area - intersection_area

        # giou : [batch_size, 13, 13, 3, 1]
        giou = (intersection_area+1e-10) / combine_area

        dX_wrt_t = -1 * (pre_rb[..., 0:1] - pre_lt[..., 0:1])
        dX_wrt_b = pre_rb[..., 0:1] - pre_lt[..., 0:1]
        dX_wrt_l = -1 * (pre_rb[..., 1:2] - pre_lt[..., 1:2])
        dX_wrt_r = pre_rb[..., 1:2] - pre_lt[..., 1:2]

        dI_wrt_t = tf.where(tf.math.greater(pre_lt[..., 1:2], truth_lt[..., 1:2]),
                                                    -1 * lw, tf.zeros_like(truth_lt[..., 1:2]))
        dI_wrt_b = tf.where(tf.math.less(pre_rb[..., 1:2], truth_rb[..., 1:2]),
                                                    lw, tf.zeros_like(truth_rb[..., 1:2]))
        dI_wrt_l = tf.where(tf.math.greater(pre_lt[..., 0:1], truth_lt[..., 0:1]),
                                                    -1 * lh, tf.zeros_like(truth_lt[..., 0:1]))
        dI_wrt_r = tf.where(tf.math.less(pre_rb[..., 0:1], truth_rb[..., 0:1]),
                                                    lh, tf.zeros_like(truth_rb[..., 0:1]))

        dU_wrt_t = dX_wrt_t - dI_wrt_t
        dU_wrt_b = dX_wrt_b - dI_wrt_b
        dU_wrt_l = dX_wrt_l - dI_wrt_l
        dU_wrt_r = dX_wrt_r - dI_wrt_r

        # dC_wrt_t = tf.where(tf.math.less(pre_lt[..., 1:2], truth_lt[..., 1:2]),
        #                                             -1 * giou_Cw, tf.zeros_like(truth_lt[..., 1:2]))
        # dC_wrt_b = tf.where(tf.math.greater(pre_rb[..., 1:2], truth_rb[..., 1:2]),
        #                                             giou_Cw, tf.zeros_like(truth_rb[..., 1:2]))
        # dC_wrt_l = tf.where(tf.less(pre_lt[..., 0:1], truth_lt[..., 0:1]),
        #                                             -1 * giou_Ch, tf.zeros_like(truth_lt[..., 0:1]))
        # dC_wrt_r = tf.where(tf.math.greater(pre_rb[..., 0:1], truth_rb[..., 0:1]),
        #                                             giou_Ch, tf.zeros_like(truth_rb[..., 0:1]))
        
        p_dt = ((U * dI_wrt_t) - (intersection_area * dU_wrt_t)) / (U * U)
        p_db = ((U * dI_wrt_b) - (intersection_area * dU_wrt_b)) / (U * U)
        p_dl = ((U * dI_wrt_l) - (intersection_area * dU_wrt_l)) / (U * U)
        p_dr = ((U * dI_wrt_r) - (intersection_area * dU_wrt_r)) / (U * U)

        p_dt = tf.where(tf.math.less(pre_lt[..., 1:2], pre_rb[..., 1:2]),
                                            p_dt, p_db)
        p_db = tf.where(tf.math.less(pre_lt[..., 1:2], pre_rb[..., 1:2]),
                                            p_db, p_dt)
        p_dl = tf.where(tf.math.less(pre_lt[..., 0:1], pre_rb[..., 0:1]),
                                            p_dl, p_dr)
        p_dr = tf.where(tf.math.less(pre_lt[..., 0:1], pre_rb[..., 0:1]),
                                            p_dr, p_dl)

        loss_x = p_dl + p_dr
        loss_y = p_dt + p_db
        loss_w = p_dr - p_dl
        loss_h = p_db - p_dt

        # 中心点距离
        # [batch_size, 13, 13, 3, 2]
        dx_dy = pre_xy - yi_true_xy
        # [batch_size, 13, 13, 3, 1]
        S = tf.square(dx_dy[..., 0:1]) + tf.square(dx_dy[..., 1:2])

        # 外围框距离
        # [batch_size, 13, 13, 3, 2]
        cx_cy = combine_right_bottom - combine_left_top
        Cw = cx_cy[..., 0:1]
        Ch = cx_cy[..., 1:2]
        # [batch_size, 13, 13, 3, 1]
        C = tf.square(Cw) + tf.square(Ch)

        dCw_dy = 0.0 
        dCh_dx = 0.0   
        dCh_dw = 0.0  
        dCw_dh = 0.0

        dCr_dx = tf.where(tf.math.greater(pre_rb[..., 0:1], truth_rb[..., 0:1]),
                                                tf.ones_like(pre_rb[..., 0:1]), 
                                                tf.zeros_like(truth_rb[..., 0:1]))
        dCl_dx = tf.where(tf.math.less(pre_lt[..., 0:1], truth_lt[..., 0:1]),
                                                tf.ones_like(pre_lt[..., 0:1]),
                                                tf.zeros_like(truth_lt[..., 0:1]))                                                
        dCw_dx = dCr_dx - dCl_dx

        dCb_dy = tf.where(tf.math.greater(pre_rb[..., 1:2], truth_rb[..., 1:2]),
                                                tf.ones_like(pre_rb[..., 1:2]),
                                                tf.zeros_like(truth_rb[..., 1:2]))
        dCt_dy = tf.where(tf.math.less(pre_lt[..., 1:2], truth_lt[..., 1:2]),
                                                tf.ones_like(pre_lt[..., 1:2]),
                                                tf.zeros_like(truth_lt[..., 1:2]))
        dCh_dy = dCb_dy - dCt_dy

        dCb_dh = dCb_dy / 2.0
        dCt_dh = dCt_dy / (-2.0)
        dCh_dh = dCb_dh - dCt_dh

        dCr_dw = dCr_dx / 2.0
        dCl_dw = dCl_dx / (-2.0)
        dCw_dw = dCr_dw - dCl_dw

        pi = 3.14159265358979323846

        ar_gt = yi_true_wh[..., 0:1] / yi_true_wh[..., 1:2]
        ar_pred = pre_wh[...,0:1] / pre_wh[..., 1:2]
        ar_loss = 4 / (pi * pi) * tf.square( tf.math.atan(ar_gt) - tf.math.atan(ar_pred) )
        alpha = ar_loss / (1.0 - iou + ar_loss + 1e-10) # iou:I/U
        ar_dh_dw = 8 / (pi*pi) * (
                                tf.math.atan(ar_gt) - tf.math.atan(ar_pred)
                            ) * pre_wh
        ar_dw = ar_dh_dw[..., 1:2]
        ar_dh = ar_dh_dw[..., 0:1]

        loss_x = tf.where(tf.math.greater(C, 0.0),
                loss_x + (2*(yi_true_xy[..., 0:1] - pre_xy[..., 0:1])*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C),
                loss_x)
        loss_y = tf.where(tf.math.greater(C, 0.0),
                loss_y + (2*(yi_true_xy[..., 1:2] - pre_xy[..., 1:2])*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C),
                loss_y)
        loss_w = tf.where(tf.math.greater(C, 0.0),
                loss_w + (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C) + alpha * ar_dw,
                loss_w)
        loss_h = tf.where(tf.greater(C, 0.0),
                loss_h + (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C) + alpha * ar_dh,
                loss_h)

        loss_x += (2*(yi_true_xy[..., 0:1] - pre_xy[..., 0:1])*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C)
        loss_y += (2*(yi_true_xy[..., 1:2] - pre_xy[..., 1:2])*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C)            
        loss_w += (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C) + alpha * ar_dw
        loss_h += (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C) + alpha * ar_dh

        # [batch_size, 13, 13, 3, 4]
        loss_xywh = tf.concat([loss_x, loss_y, loss_w, loss_h], -1)
        return loss_xywh

    # 得到低iou的地方
    def __get_low_iou_mask(self, pre_xy, pre_wh, yi_true, use_iou=True, ignore_thresh=0.5):
        '''
        pre_xy:[batch_size, 13, 13, 3, 2]
        pre_wh:[batch_size, 13, 13, 3, 2]
        yi_true:[batch_size, 13, 13, 3, 5+class_num]
        use_iou:是否使用 iou 作为计算标准
        ignore_thresh:iou小于这个值就认为与真值不重合
        return: [batch_size, 13, 13, 3, 1]
        返回 iou 低于阈值的区域 mask
        '''
        # 置信度:[batch_size, 13, 13, 3, 1]
        conf_yi_true = yi_true[..., 4:5]

        # iou小的地方
        low_iou_mask = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        # batch_size
        N = tf.shape(yi_true)[0]
        
        def loop_cond(index, low_iou_mask):
            return tf.less(index, N)        
        def loop_body(index, low_iou_mask):
            # 一张图片的全部真值 : [13, 13, 3, class_num+5] & [13, 13, 3, 1] == > [V, class_num + 5]
            valid_yi_true = tf.boolean_mask(yi_true[index], tf.cast(conf_yi_true[index, ..., 0], tf.bool))
            # 计算 iou/ giou : [13, 13, 3, V]
            iou, giou = self.IOU(pre_xy[index], pre_wh[index], valid_yi_true)

            # [13, 13, 3]
            if use_iou:
                best_giou = tf.reduce_max(iou, axis=-1)
            else:
                best_giou = tf.reduce_max(giou, axis=-1)
            # [13, 13, 3]
            low_iou_mask_tmp = best_giou < ignore_thresh
            # [13, 13, 3, 1]
            low_iou_mask_tmp = tf.expand_dims(low_iou_mask_tmp, -1)
            # 写入
            low_iou_mask = low_iou_mask.write(index, low_iou_mask_tmp)
            return index+1, low_iou_mask

        _, low_iou_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, low_iou_mask])
        # 拼接:[batch_size, 13, 13, 3, 1]
        low_iou_mask = low_iou_mask.stack()
        return low_iou_mask

    # 得到预测物体概率低的地方的掩码
    def __get_low_prob_mask(self, prob, prob_thresh=0.25):
        '''
        prob:[batch_size, 13, 13, 3, class_num]
        prob_thresh:物体概率预测的阈值
        return: bool [batch_size, 13, 13, 3, 1]
        全部预测物体概率小于阈值的 mask
        '''
        # [batch_size, 13, 13, 3, 1]
        max_prob = tf.reduce_max(prob, axis=-1, keepdims=True)
        low_prob_mask = max_prob < prob_thresh        
        return low_prob_mask

    # 对预测值进行解码
    def __decode_feature(self, yi_pred, curr_anchors):
        '''
        yi_pred:[batch_size, 13, 13, 3 * (class_num + 5)]
        curr_anchors:[3,2], 这一层对应的 anchor, 真实值
        return:
            xy:[batch_size, 13, 13, 3, 2], 相对于原图的中心坐标
            wh:[batch_size, 13, 13, 3, 2], 相对于原图的宽高
            conf:[batch_size, 13, 13, 3, 1]
            prob:[batch_size, 13, 13, 3, class_num]
        '''
        shape = tf.shape(yi_pred) 
        shape = tf.cast(shape, tf.float32)
        # [batch_size, 13, 13, 3, class_num + 5]
        yi_pred = tf.reshape(yi_pred, [shape[0], shape[1], shape[2], 3, 5 + self.class_num])
        # 分割预测值
        # shape : [batch_size,13,13,3,2] [batch_size,13,13,3,2] [batch_size,13,13,3,1] [batch_size,13,13,3, class_num]
        xy, wh, conf, prob = tf.split(yi_pred, [2, 2, 1, self.class_num], axis=-1)

        ''' 计算 x,y 的坐标偏移 '''
        offset_x = tf.range(shape[2], dtype=tf.float32) #宽
        offset_y = tf.range(shape[1], dtype=tf.float32) # 高
        offset_x, offset_y = tf.meshgrid(offset_x, offset_y)
        offset_x = tf.reshape(offset_x, (-1, 1))
        offset_y = tf.reshape(offset_y, (-1, 1))
        # 把 offset_x, offset_y 合并成一个坐标网格 [13*13, 2], 每个元素是当前坐标 (x, y)
        offset_xy = tf.concat([offset_x, offset_y], axis=-1)
        # [13, 13, 1, 2]
        offset_xy = tf.reshape(offset_xy, [shape[1], shape[2], 1, 2])
        
        xy = tf.math.sigmoid(xy) + offset_xy    
        xy = xy / [shape[2], shape[1]]

        wh = tf.math.exp(wh) * curr_anchors
        wh = wh / [self.width, self.height]

        return xy, wh, conf, prob

    # 计算损失, yolov3
    def __compute_loss_v3(self, xy, wh, conf, prob, yi_true, low_iou_mask):
        '''
        xy:[batch_size, 13, 13, 3, 2]
        wh:[batch_size, 13, 13, 3, 2]
        conf:[batch_size, 13, 13, 3, 1]
        prob:[batch_size, 13, 13, 3, class_num]
        yi_true:[batch_size, 13, 13, 3, class_num]
        low_iou_mask:[batch_size, 13, 13, 3, 1]
        return: 总的损失

        loss_total:总的损失
        xy_loss:中心坐标损失
        wh_loss:宽高损失
        conf_loss:置信度损失
        class_loss:分类损失
        '''
        # bool => float32
        low_iou_mask = tf.cast(low_iou_mask, tf.float32)
        # batch_size
        N = tf.shape(xy)[0]
        N = tf.cast(N, tf.float32)

        # [batch_size, 13, 13, 3, 1]
        no_obj_mask = 1.0 - yi_true[..., 4:5]
        conf_loss_no_obj = no_obj_mask * low_iou_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=yi_true[:,:,:,:,4:5], logits=conf)

        # [batch_size, 13, 13, 3, 1]
        obj_mask = yi_true[..., 4:5]
        conf_loss_obj = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=yi_true[:,:,:,:,4:5], logits=conf)
        
        # 置信度损失
        conf_loss = conf_loss_obj + conf_loss_no_obj
        conf_loss = tf.reduce_sum(conf_loss) / N
        
        # 平衡系数
        loss_scale = tf.square(2. - yi_true[..., 2:3] * yi_true[..., 3:4])

        # xy 损失
        xy_loss = loss_scale * obj_mask * tf.square(yi_true[..., 0: 2] - xy)
        xy_loss = tf.reduce_sum(xy_loss) / N

        # wh 损失
        wh_y_true = tf.where(condition=tf.equal(yi_true[..., 2:4], 0),
                                        x=tf.ones_like(yi_true[..., 2: 4]), y=yi_true[..., 2: 4])
        wh_y_pred = tf.where(condition=tf.equal(wh, 0),
                                        x=tf.ones_like(wh), y=wh)
        wh_y_true = tf.clip_by_value(wh_y_true, 1e-10, 1e10)
        wh_y_pred = tf.clip_by_value(wh_y_pred, 1e-10, 1e10)
        wh_y_true = tf.math.log(wh_y_true)
        wh_y_pred = tf.math.log(wh_y_pred)

        wh_loss = loss_scale * obj_mask * tf.square(wh_y_true - wh_y_pred)
        wh_loss = tf.reduce_sum(wh_loss) / N

        # prob 损失
        class_loss = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=yi_true[...,5:5+self.class_num], logits=prob)
        class_loss = tf.reduce_sum(class_loss) / N

        loss_total = xy_loss + wh_loss + conf_loss + class_loss
        return loss_total
        
    # 计算损失, yolov4
    def __compute_loss_v4(self, xy, wh, conf, prob, yi_true, cls_normalizer=1.0, ignore_thresh=0.5, 
                                                                prob_thresh=0.25, score_thresh=0.25, iou_normalizer=0.07):
        '''
        xy:[batch_size, 13, 13, 3, 2]
        wh:[batch_size, 13, 13, 3, 2]
        conf:[batch_size, 13, 13, 3, 1]
        prob:[batch_size, 13, 13, 3, class_num]
        yi_true:[batch_size, 13, 13, 3, class_num]
        cls_normalizer:置信度损失参数
        ignore_thresh:与真值iou阈值
        prob_thresh:最低分类概率阈值
        score_thresh:最低分类得分阈值
        iou_normalizer:iou_loss 系数
        return: 总的损失

        loss_total:总的损失
        xy_loss:中心坐标损失
        wh_loss:宽高损失
        conf_loss:置信度损失
        class_loss:分类损失
        '''
        # mask of low iou 
        low_iou_mask = self.__get_low_iou_mask(xy, wh, yi_true, ignore_thresh=ignore_thresh)
        # mask of low prob
        low_prob_mask = self.__get_low_prob_mask(prob, prob_thresh=prob_thresh)        
        # mask of low iou or low prob
        low_iou_prob_mask = tf.math.logical_or(low_iou_mask, low_prob_mask)
        low_iou_prob_mask = tf.cast(low_iou_prob_mask, tf.float32)

        # batch_size
        N = tf.shape(xy)[0]
        N = tf.cast(N, tf.float32)

        # [batch_size, 13, 13, 3, 1]
        conf_scale = wh[..., 0:1] * wh[..., 1:2]
        conf_scale = tf.where(tf.math.greater(conf_scale, 0),
                                                        tf.math.sqrt(conf_scale), conf_scale)
        conf_scale = conf_scale * cls_normalizer                                                        
        conf_scale = tf.math.square(conf_scale)
        # [batch_size, 13, 13, 3, 1]
        no_obj_mask = 1.0 - yi_true[..., 4:5]
        conf_loss_no_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                            labels=yi_true[:,:,:,:,4:5], logits=conf
                                                            ) * conf_scale * no_obj_mask * low_iou_prob_mask
        # [batch_size, 13, 13, 3, 1]
        obj_mask = yi_true[..., 4:5]        
        conf_loss_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                            labels=yi_true[:,:,:,:,4:5], logits=conf
                                                            ) * np.square(cls_normalizer) * obj_mask        
        # 置信度损失
        conf_loss = conf_loss_obj + conf_loss_no_obj
        conf_loss = tf.reduce_sum(conf_loss) / N

        # giou_loss
        giou_loss = self.__my_GIOU_loss(xy, wh, yi_true[..., 0:4])
        # giou_loss = self.__my_CIOU_loss(xy, wh, yi_true[..., 0:4])
        giou_loss = tf.clip_by_value(giou_loss, 1e-10, 1e10)
        giou_loss = tf.square(giou_loss * obj_mask) * iou_normalizer
        giou_loss = tf.reduce_sum(giou_loss) / N

        # xy 损失
        xy_loss = obj_mask * tf.square(yi_true[..., 0: 2] - xy)
        xy_loss = tf.reduce_sum(xy_loss) / N

        # wh 损失
        wh_y_true = tf.where(condition=tf.equal(yi_true[..., 2:4], 0),
                                        x=tf.ones_like(yi_true[..., 2: 4]), y=yi_true[..., 2: 4])
        wh_y_pred = tf.where(condition=tf.equal(wh, 0),
                                        x=tf.ones_like(wh), y=wh)
        wh_y_true = tf.clip_by_value(wh_y_true, 1e-10, 1e10)
        wh_y_pred = tf.clip_by_value(wh_y_pred, 1e-10, 1e10)
        wh_y_true = tf.math.log(wh_y_true)
        wh_y_pred = tf.math.log(wh_y_pred)

        wh_loss = obj_mask * tf.square(wh_y_true - wh_y_pred)
        wh_loss = tf.reduce_sum(wh_loss) / N
        
        # prob 损失
        # [batch_size, 13, 13, 3, class_num]
        prob_score = prob * conf
        
        high_score_mask = prob_score > score_thresh
        high_score_mask = tf.cast(high_score_mask, tf.float32)
        
        class_loss_no_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                        labels=yi_true[...,5:5+self.class_num],
                                                        logits=prob 
                                                    ) * low_iou_prob_mask * no_obj_mask * high_score_mask
        
        class_loss_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                        labels=yi_true[...,5:5+self.class_num],
                                                        logits=prob
                                                    ) * obj_mask

        class_loss = class_loss_no_obj + class_loss_obj        
        class_loss = tf.reduce_sum(class_loss) / N

        loss_total = xy_loss + wh_loss + conf_loss + class_loss + giou_loss
        # loss_total = giou_loss + conf_loss + class_loss
        return loss_total

    # 获得损失 yolov3
    def get_loss(self, feature_y1, feature_y2, feature_y3, y1_true, y2_true, y3_true, use_iou=True, ignore_thresh=0.5):
        '''
        feature_y1:[batch_size, 13, 13, 3*(5+class_num)]
        feature_y2:[batch_size, 26, 26, 3*(5+class_num)]
        feature_y3:[batch_size, 52, 52, 3*(5+class_num)]
        return : loss_total, loss_xy, loss_wh, loss_conf, loss_class
        use_iou:bool, 使用 iou 作为与真值匹配的衡量标准, 否则使用 giou 
        ignore_thresh    # iou低于这个数就认为与真值不重合
        return:total_loss
        '''
        # y1
        xy, wh, conf, prob = self.__decode_feature(feature_y1, self.anchors[2])
        low_iou_mask_y1 = self.__get_low_iou_mask(xy, wh, y1_true, use_iou=use_iou, ignore_thresh=ignore_thresh)
        loss_y1 = self.__compute_loss_v3(xy, wh, conf, prob, y1_true, low_iou_mask_y1)

        # y2
        xy, wh, conf, prob = self.__decode_feature(feature_y2, self.anchors[1])
        low_iou_mask_y2 = self.__get_low_iou_mask(xy, wh, y2_true, use_iou=use_iou, ignore_thresh=ignore_thresh)
        loss_y2 = self.__compute_loss_v3(xy, wh, conf, prob, y2_true, low_iou_mask_y2)

        # y3
        xy, wh, conf, prob = self.__decode_feature(feature_y3, self.anchors[0])
        low_iou_mask_y3 = self.__get_low_iou_mask(xy, wh, y3_true, use_iou=use_iou, ignore_thresh=ignore_thresh)
        loss_y3 = self.__compute_loss_v3(xy, wh, conf, prob, y3_true, low_iou_mask_y3)

        return loss_y1 + loss_y2 + loss_y3


    # 获得损失 yolov4
    def get_loss_v4(self, feature_y1, feature_y2, feature_y3, y1_true, y2_true, y3_true, cls_normalizer=1.0, ignore_thresh=0.5, prob_thresh=0.25, score_thresh=0.25):
        '''
        feature_y1:[batch_size, 13, 13, 3*(5+class_num)]
        feature_y2:[batch_size, 26, 26, 3*(5+class_num)]
        feature_y3:[batch_size, 52, 52, 3*(5+class_num)]
        y1_true: y1尺度的
        y2_true: y2尺度的标签
        y3_true: y3尺度的标签
        cls_normalizer:分类损失系数
        ignore_thresh:与真值 iou 阈值
        prob_thresh:分类概率最小值
        score_thresh:分类得分最小值
        return:total_loss
        '''
        # y1
        xy, wh, conf, prob = self.__decode_feature(feature_y1, self.anchors[2])
        loss_y1 = self.__compute_loss_v4(xy, wh, conf, prob, y1_true, cls_normalizer=1.0, 
                                                                                    ignore_thresh=ignore_thresh, prob_thresh=0.25, score_thresh=0.25)

        # y2
        xy, wh, conf, prob = self.__decode_feature(feature_y2, self.anchors[1])
        loss_y2 = self.__compute_loss_v4(xy, wh, conf, prob, y2_true, cls_normalizer=1.0, 
                                                                                    ignore_thresh=ignore_thresh, prob_thresh=0.25, score_thresh=0.25)

        # y3
        xy, wh, conf, prob = self.__decode_feature(feature_y3, self.anchors[0])
        loss_y3 = self.__compute_loss_v4(xy, wh, conf, prob, y3_true, cls_normalizer=1.0, 
                                                                                    ignore_thresh=ignore_thresh, prob_thresh=0.25, score_thresh=0.25)

        return loss_y1 + loss_y2 + loss_y3


    # 非极大值抑制
    def __nms(self, boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_threshold=0.5):
        '''
        boxes:[1, V, 4]
        score:[1, V, class_num]
        num_classes:分类数
        max_boxes:一类最大保留多少个 box
        score_thresh:小于这个分数的不算
        iou_threshold:iou大于这个的合并
        return:????
            boxes:[V, 4]
            score:[V,]
        '''
        boxes_list, label_list, score_list = [], [], []
        max_boxes = tf.constant(max_boxes, dtype='int32')

        # [V, 4]
        boxes = tf.reshape(boxes, [-1, 4])
        # [V, class_num]
        score = tf.reshape(scores, [-1, num_classes])

        # 分数大的掩码
        mask = tf.greater_equal(score, tf.constant(score_thresh))
        # 对每一个分类进行 nms 操作
        for i in range(num_classes):
            # Step 3: Apply the mask to scores, boxes and pick them out
            filter_boxes = tf.boolean_mask(boxes, mask[:,i])
            filter_score = tf.boolean_mask(score[:,i], mask[:,i])
            nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                    scores=filter_score,
                                                    max_output_size=max_boxes,
                                                    iou_threshold=iou_threshold, name='nms_indices')
            label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
            boxes_list.append(tf.gather(filter_boxes, nms_indices))
            score_list.append(tf.gather(filter_score, nms_indices))

        # 合并
        boxes = tf.concat(boxes_list, axis=0)
        score = tf.concat(score_list, axis=0)
        label = tf.concat(label_list, axis=0)

        return boxes, score, label

    # 得到预测的全部 box
    def __get_pred_box(self, feature_y1, feature_y2, feature_y3):
        '''
        feature_y1:[1, 13, 13, 3*(class_num + 5)]
        feature_y1:[1, 26, 52, 3*(class_num + 5)]
        feature_y1:[1, 26, 52, 3*(class_num + 5)]
        return:
            boxes:[1, V, 4]:[x_min, y_min, x_max, y_max] 相对于原始图片大小的浮点数
            conf:[1, V, 1]
            prob:[1, V, class_num]
        '''
        # y1解码
        xy1, wh1, conf1, prob1 = self.__decode_feature(feature_y1, self.anchors[2])
        conf1, prob1 = tf.sigmoid(conf1), tf.sigmoid(prob1)

        # y2解码
        xy2, wh2, conf2, prob2 = self.__decode_feature(feature_y2, self.anchors[1])
        conf2, prob2 = tf.sigmoid(conf2), tf.sigmoid(prob2)

        # y3解码
        xy3, wh3, conf3, prob3 = self.__decode_feature(feature_y3, self.anchors[0])
        conf3, prob3 = tf.sigmoid(conf3), tf.sigmoid(prob3)

        # 把 box 放到一起来
        def _reshape(xy, wh, conf, prob):
            # [1, 13, 13, 3, 1]
            x_min = xy[..., 0: 1] - wh[..., 0: 1] / 2.0
            x_max = xy[..., 0: 1] + wh[..., 0: 1] / 2.0
            y_min = xy[..., 1: 2] - wh[..., 1: 2] / 2.0
            y_max = xy[..., 1: 2] + wh[..., 1: 2] / 2.0

            # [1, 13, 13, 3, 4]
            boxes = tf.concat([x_min, y_min, x_max, y_max], -1)
            shape = tf.shape(boxes)
            # [1, 13*13*3, 4]
            boxes = tf.reshape(boxes, (shape[0], shape[1] * shape[2]* shape[3], shape[4]))

            # [1, 13 * 13 * 3, 1]
            conf = tf.reshape(conf, (shape[0], shape[1] * shape[2]* shape[3], 1))

            # [1, 13 * 13 * 3, class_num]
            prob = tf.reshape(prob, (shape[0], shape[1] * shape[2]* shape[3], -1))
        
            return boxes, conf, prob

        # reshape
        # [batch_size, 13*13*3, 4], [batch_size, 13*13*3, 1], [batch_size, 13*13*3, class_num]
        boxes_y1, conf_y1, prob_y1 = _reshape(xy1, wh1, conf1, prob1)
        boxes_y2, conf_y2, prob_y2 = _reshape(xy2, wh2, conf2, prob2)
        boxes_y3, conf_y3, prob_y3 = _reshape(xy3, wh3, conf3, prob3)

        # 全部拿到一起来
        # [1, 13*13*3, 4] & [1, 26*26*3, 4] & [1, 52*52*3, 4] ==> [1,  V, 4]
        boxes = tf.concat([boxes_y1, boxes_y2, boxes_y3], 1)
        conf = tf.concat([conf_y1, conf_y2, conf_y3], 1)
        prob = tf.concat([prob_y1, prob_y2, prob_y3], 1)

        return boxes, conf, prob

    # 得到预测结果
    def get_predict_result(self, feature_y1, feature_y2, feature_y3, class_num, score_thresh=0.5, iou_thresh=0.5, max_box=200):
        '''
        feature_y1:[batch_size, 13, 13, 3*(class_num+5)]
        feature_y2:[batch_size, 26, 26, 3*(class_num+5)]
        feature_y3:[batch_size, 52, 52, 3*(class_num+5)]
        class_num:分类数
        score_thresh:小于这个分数的就不算
        iou_thresh : 超过这个 iou 的 box 进行合并
        max_box : 最多保留多少物体
        return:
            boxes:[V, 4]包含[x_min, y_min, x_max, y_max]
            score:[V, 1]
            label:[V, 1]
        '''
        boxes, conf, prob = self.__get_pred_box(feature_y1, feature_y2, feature_y3)
        pre_score = conf * prob
        boxes, score, label = self.__nms(boxes, pre_score, class_num, max_boxes=max_box, score_thresh=score_thresh, iou_threshold=iou_thresh)
        return boxes, score, label

