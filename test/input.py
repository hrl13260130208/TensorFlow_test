#coding=utf-8

import os
import numpy as np
import tensorflow as tf
import PIL

path="d:/dataset/train/"

cars=[]
label_cars=[]

def get_files(file_dir):
    for file in os.listdir(path):
        if os.path.splitext( file )[1] == '.jpg':
            cars.append(path+file)
            label_cars.append(1)

get_files(path)
temp=np.array([cars,label_cars])
temp=temp.transpose()
np.random.shuffle(temp)
print("temp的值为：",temp)

image=list(temp[:,0])
label=list(temp[:,1])

def get_batch(image,label,image_W,image_H,batch_size,capacity):

    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.int32)

    input_queue=tf.train.slice_input_producer([image,label])

    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])

    image=tf.image.decode_jpeg(image_contents,channels=3)

    image=tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image=tf.image.per_image_standardization(image)

    image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,
                                               num_threads=32,capacity=capacity)

    label_batch=tf.reshape(label_batch,[batch_size])
    image_batch=tf.cast(image_batch,tf.float32)
    return image_batch,label_batch

BATCH_SIZE=2
CAPACITY=256
IMAGE_W=1024
IMAGE_H=768

image_batch,label_batch=get_batch(image,label,IMAGE_W,IMAGE_H,BATCH_SIZE,CAPACITY)


X=tf.placeholder(tf.float32,[None,1024*768])
W=tf.Variable(tf.zeros([1024*768,1]) )
b=tf.Variable(tf.zeros([1]))
y=tf.nn.softmax(tf.matmul(X,W)+b)

y_=tf.placeholder(tf.float32,[None,1])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

print(image_batch)
print(label_batch)
print("strat!")

coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(coord=coord)

try:
    img,lab=sess.run([image_batch,label_batch])
    print( "img的值：\n", img )
    print( "lab的值：\n", lab )
except :
    print("file finish!")


print("stop!")
#for _ in range(200) :

   # sess.run(train_step, feed_dict={X: img, y_: lab})


"""
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    sess.run( tf.initialize_all_variables() )
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners( coord=coord, sess=sess )
    for i in range( 10 ):
        val, l = sess.run( [image_batch, label_batch] )
        print( l )
    print( "complete ..." )
    coord.request_stop()
    coord.join( threads )
    sess.close()
    
    """
'''
with tf.Session() as sess:
    i=0
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and i<2:
            img,labels=sess.run([image_batch,label_batch])

            for j in np.arange(BATCH_SIZE):
                print("label:%d"%label[j])
                print("-------------"+img[j])
            for x in img:
                print(x)
            i+=1


    finally:
        coord.request_stop()

   # coord.join(threads)
'''