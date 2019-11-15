
import tensorflow as tf
import numpy as np


def t1():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
    # the model will take as input an integer matrix of size (batch,input_length).
    # the largest integer (i.e. word index) in the input should be no larger than 999(vocabularysize).
    # now model.output_shape == (None, 10, 64), where None is the batch dimension.

    input_array = np.random.randint(1000, size=(32, 10))

    model.compile('rmsprop', 'mse')
    output_array = model.predict(input_array)
    print(output_array)
    # assert output_array.shape == (32, 10, 64)


if __name__ == '__main__':
    # nums=[1,2,3,4,5,6,7,8,9]
    # nums=[[1,2],[3,2],[3,4]]
    # nums=[[1,2,2],[3,2,2],[3,4,2]]
    # nums=[[[1,2,2],[3,2,2]],[[3,4,2]]]
    #
    # n_nums=np.array(nums)
    # print(n_nums,n_nums.shape)
    input_eval = [[2,3]]
    input_eval = tf.expand_dims(input_eval, 0)
    print(input_eval,input_eval.shape)
    # text="fasdfsdfjaskdfjoiwweournvajfjfsdgh"
    # vocab = sorted(set(text))
    # print("---------------", repr(vocab))
    #
    # char2idx = {u: i for i, u in enumerate(vocab)}
    #
    # idx2char = np.array(vocab)

    # text_as_int = np.array([char2idx[c] for c in text])
    # print(text_as_int,text_as_int.shape)
    #
    # print(char2idx)
    # print(text_as_int,text_as_int.shape)
    # chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(10 + 1, drop_remainder=True)




    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Embedding(vocab.__len__(), 256, input_length=10))
    # print(model.input_shape,model.output_shape)
    # model.compile('rmsprop', 'mse')
    # print("input:\n",np.array([char2idx["h"]]).shape,np.array([char2idx["h"]]))
    # output_array = model.predict(np.array([char2idx["h"]]))
    # print("output:\n",output_array.shape,output_array)

