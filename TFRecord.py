#coding=utf-8

import os
import tensorflow as tf

#路径
path="d:/dataset/train/"
#人为设定的类
classes={'car'}
#要生成的文件
writer=tf.python_io.TFRecordWriter("car_train.tfrecords")

for index,name in enumerate(classes):

    for image_name in os.walk(path) :
        image_path=path+image_name
        image=tf.image()
        image_raw=image.resize()