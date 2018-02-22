# coding=utf-8

import os
import tensorflow as tf


def file_name(file_dir):
    '''
    :param file_dir: 文件目录 （其下图片为jpg格式，label为.json文件）
    :return:
        包含图片路径与label路径
    '''
    images_name = []
    labels_name = []
    for root, dirs, files in os.walk( file_dir ):
        for file in files:
            if os.path.splitext( file )[1] == '.jpg':
                images_name.append( os.path.join( root, file ) )
                labels_name.append( 1 )
    print( images_name )
    print( labels_name )
    return images_name, labels_name


image_name, labels_name = file_name( "d:/dataset/train/" )
'''
读取数据到TensorFlow
'''
file_queue = tf.train.string_input_producer( image_name, shuffle=True, num_epochs=3 )
image_reader = tf.WholeFileReader()
key, image = image_reader.read( file_queue )
image = tf.image.decode_jpeg( image )
# image=tf.read_file(image_name)
print( image )
# img=tf.image.decode_jpeg(image,channels=3)
# print(img)

'''
filename=os.path.join("d:/dataset/train/1d6aa7609bc53347185de80a7c4c9ac1.jpg")
print(filename)
fid=open(filename)
print(fid)
content=fid.read()
print("read（）的结果："+content)
content=content.split('\n')
print("\n处理后的结果："+content)
content=content[-1]
print("[-1]处理后的结果："+content)
valuequeue=tf.train.string_input_producer(content,shuffle=True)
print("valuequeue:"+valuequeue)
value=valuequeue.dequeue()
print("value的值："+value)
dir,lables=tf.decode_csv(records=value,record_defaults=[["string"],[""]],field_delim="")

lables=tf.string_to_number(lables,tf.int32)
print("lables的值："+lables)

imagecontent=tf.read_file(dir)
print("imagecontent:\n"+imagecontent)

image=tf.image.decode_jpeg(imagecontent)
'''
