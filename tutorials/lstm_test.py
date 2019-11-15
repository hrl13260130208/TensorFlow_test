# coding: UTF-8
"""
    @author: samuel ko
    @date: 2018/12/12
    @link: https://blog.csdn.net/zwqjoy/article/details/80493341
"""
import numpy
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM, SimpleRNN
# from keras.utils import np_utils

import tensorflow as tf

# fix random seed for reproducibility
def train():
    numpy.random.seed(5)

    # define the raw dataset
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print(len(alphabet))
    # create mapping of characters to integers (0-25) and the reverse
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))


    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 3
    dataX = []
    dataY = []
    for i in range(0, len(alphabet) - seq_length, 1):
        seq_in = alphabet[i:i + seq_length]
        seq_out = alphabet[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
        print(seq_in, '->', seq_out)
    # 我们运行上面的代码，来观察现在我们的input和output数据集是这样一种情况
    # A -> B
    # B -> C
    # ...
    # Y -> Z

    # 喂入网络的特征为 [batch_size, time_step, input_dim] 3D的Tensor
    # 用易懂的语言就是: time_step为时间步的个数, input_dim为每个时间步喂入的数据
    X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
    # print(X)
    # [[[ 0]]
    #  [[ 1]]
    #  [[ 2]]
    #  [[ 3]]
    #  ...
    #  [[24]]]
    # normalize 最后接一个分类的任务
    X = X / float(len(alphabet))

    y_label=numpy.asarray(dataY)
    num_labels=y_label.shape[0]
    num_class=26
    offset=numpy.arange(num_labels)*num_class
    print(offset)
    label_one_hat=numpy.zeros((num_labels,num_class))
    label_one_hat.flat[offset+y_label.ravel()]=1


    # (25, 3, 1)
    # one hot编码输出label
    # y = tf.keras.np_utils.to_categorical(dataY)
    # print(y.shape)
    #
    # 创建&训练&保存模型
    print(X)
    print(X.shape)
    print(label_one_hat)
    print(label_one_hat.shape)
    model = tf.keras.Sequential()

    # input_shape = (time_step, 每个时间步的input_dim)
    model.add(tf.keras.layers.LSTM(5, input_shape=(X.shape[1], X.shape[2])))
    model.add(tf.keras.layers.Dense(label_one_hat.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, label_one_hat, nb_epoch=100, batch_size=100, verbose=2)
    model.save("simplelstm.h5")

if __name__ == '__main__':
    train()
    # model = tf.keras.models.load_model("simplelstm.h5")
    # layer_1 = tf.keras.backend.function([model.layers[0].input], [
    #     model.layers[0].output])  # 第一个 model.layers[0],不修改,表示输入数据；第二个model.layers[you wanted],修改为你需要输出的层数的编号
    # layer_11 = tf.keras.backend.function([model.layers[0].input], [
    #     model.layers[1].output])  # 第一个 model.layers[0],不修改,表示输入数据；第二个model.layers[you wanted],修改为你需要输出的层数的编号
    #
    # # 定义shape为(1, 3, 1)的输入，输入网络
    # inputs = numpy.array([[0], [0.03846154], [0.07692308]])
    # inputs = numpy.expand_dims(inputs, 0)
    #
    # print(layer_1([inputs])[0])
    # print(layer_1([inputs])[0].shape)
    # print(layer_11([inputs])[0])
    # print(layer_11([inputs])[0].shape)
    # print(model.predict(inputs))

