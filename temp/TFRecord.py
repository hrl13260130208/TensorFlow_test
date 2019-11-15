#coding=utf-8

import os
import tensorflow as tf

# #路径
# path="d:/dataset/train/"
# #人为设定的类
# classes={'car'}
# #要生成的文件
# writer=tf.python_io.TFRecordWriter("car_train.tfrecords")
#
# for index,name in enumerate(classes):
#
#     for image_name in os.walk(path) :
#         image_path=path+image_name
#         image=tf.image()
#         image_raw=image.resize()
#         writer.write()

import sys
import math
import os
import tensorflow as tf

def convert_dataset(list_path, data_dir, output_dir, _NUM_SHARDS=5):
    fd = open(list_path)
    lines = [line.split() for line in fd]
    fd.close()
    num_per_shard = int(math.ceil(len(lines) / float(_NUM_SHARDS)))
    with tf.Graph().as_default():
        decode_jpeg_data = tf.placeholder(dtype=tf.string)
        decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_path = os.path.join(output_dir,
                    'data_{:05}-of-{:05}.tfrecord'.format(shard_id, _NUM_SHARDS))
                tfrecord_writer = tf.python_io.TFRecordWriter(output_path)
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(lines))
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Converting image {}/{} shard {}'.format(
                        i + 1, len(lines), shard_id))
                    sys.stdout.flush()
                    image_data = tf.gfile.FastGFile(os.path.join(data_dir, lines[i][0]), 'rb').read()
                    image = sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
                    height, width = image.shape[0], image.shape[1]
                    example = dataset_utils.image_to_tfexample(
                        image_data, b'jpg', height, width, int(lines[i][1]))
                    tfrecord_writer.write(example.SerializeToString())
                tfrecord_writer.close()
    sys.stdout.write('\n')
    sys.stdout.flush()

os.system('mkdir -p train')
convert_dataset('list_train.txt', 'flower_photos', 'train/')
os.system('mkdir -p val')
convert_dataset('list_val.txt', 'flower_photos', 'val/')


