# coding:utf-8
# 激活函数的实现
from enum import Enum
import tensorflow as tf

class activation(Enum):
    MISH = 1        # mish 激活
    LEAKY_RELU = 2  # leaky_relu
    RELU = 3        # relu 激活

def activation_fn(inputs, name, alpha=0.1):
    if name is activation.MISH:        
        MISH_THRESH = 20.0        # 阈值
        tmp = inputs

        inputs = tf.where(
                                    tf.math.logical_and(tf.less(inputs, MISH_THRESH), tf.greater(inputs, -MISH_THRESH)),
                                    tf.log(1 + tf.exp(inputs)), 
                                    tf.zeros_like(inputs)
                                )
        inputs = tf.where(tf.less(inputs, -MISH_THRESH), 
                                                tf.exp(inputs), 
                                                inputs)
        # Mish = x*tanh(ln(1+e^x))
        inputs = tmp * tf.tanh(inputs)
        return inputs
    elif name is activation.LEAKY_RELU:
        return tf.nn.leaky_relu(inputs, alpha=alpha)
    elif name is activation.RELU:
        return tf.nn.relu(inputs)
    elif name is None:
        return inputs
    else:
        ValueError("没有激活函数为'"+str(name) + "'")
    return None