
import tensorflow as tf
import numpy as np
import time
import os



class Model(tf.keras.Model):
  def __init__(self, units):
    super(Model, self).__init__()
    self.units = units

    self.lstm = tf.keras.layers.LSTM(self.units,
                                 return_sequences=True,
                                 recurrent_activation='sigmoid',
                                 recurrent_initializer='glorot_uniform',
                                 stateful=True)

    self.fc = tf.keras.layers.Dense(units)

  def call(self, x):

    output = self.lstm(x)
    print("===============",output.shape)

    prediction = self.fc(output)

    return prediction


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

optimizer = tf.train.AdamOptimizer()

# Using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)

checkpoint_dir = r'C:\data\rnn\stock'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

if __name__ == '__main__':
    np_data=np.load(r"C:\File\numpy_data\1.npy")
    print(np_data)
    # input1=np.array([np_data[1]])
    # output1=np.array([np_data[0]])
    # output1=output1.reshape(1,10)
    # print(input1.shape)
    # print(output1.shape)
    #
    # seq_length = 10
    # model=tf.keras.Sequential()
    # # model.build((10,1))
    # model.add(tf.keras.layers.LSTM(10, input_shape=(input1.shape[1], input1.shape[2]), recurrent_activation='sigmoid'))
    # # model.add(tf.keras.layers.Dense(10, activation='softmax'))
    #
    # model.compile(optimizer=tf.train.AdamOptimizer(0.001),loss=tf.keras.losses.categorical_crossentropy,
    #           metrics=[tf.keras.metrics.categorical_accuracy])
    #
    # model.fit(input1,output1,epochs=10000,batch_size=1)





