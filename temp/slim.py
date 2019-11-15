
#coding=utf-8

from __future__ import  absolute_import
from __future__ import  division
from __future__ import  print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys


def cifarnet(images,num_classes=10,
             is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='CifarNet'):
    '''
    创建CifarNeet模型
    :param images:  输入图像 形状[n.h,w,c]
    :param num_classes: 类别数
    :param is_training:  是否训练 模型训练设置为True，测试、推理设置为False
    :param dropout_keep_prob: dropout保持效率
    :param prediction_fn: 输出层激活函数
    :param scope:节点名
    :return:
         net：2D Tensor ,logits （pre-softmax激活）如果num_classes
            是非零整数，或者如果num_classes为0或None输入到逻辑层
        end_points：从网络组件到相应的字典激活。
    '''

