from tensorflow.examples.tutorials.mnist import input_data

import  tensorflow as tf
import numpy as np

print("start")
mnist = input_data.read_data_sets("mnist/", one_hot=True)
print("go on")

X=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]) )
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(X,W)+b)

y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)
sess=tf.InteractiveSession()
#tf.global_variables_initializer().run()

saver=tf.train.Saver()

saver.restore(sess,"data/testdata.ckpt")
for _ in range(200) :
    batch_xs , batch_ys=mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X: batch_xs, y_: batch_ys})

saver_path=saver.save(sess,"data/testdata.ckpt")

print("Model saved in file:",saver_path)
correct_prediction =tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

print(
    sess.run(accuracy, feed_dict={X:mnist.test.images,y_:mnist.test.labels})
)






