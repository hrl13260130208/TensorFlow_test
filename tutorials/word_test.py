
import tensorflow as tf
import numpy




def create_data():
    pass

class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Model, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,input_shape=(None,3))
        self.lstm = tf.keras.layers.LSTM(units)
        self.fc = tf.keras.layers.Dense(vocab_size)


    def call(self,x):
        embedding = self.embedding(x)

        print(embedding.shape)
        # output at every time step
        # output shape == (batch_size, seq_length, hidden_size)
        output = self.lstm(embedding)
        print("===============",output.shape)
        # The dense layer will output predictions for every time_steps(seq_length)
        # output shape after the dense layer == (seq_length * batch_size, vocab_size)
        prediction = self.fc(output)

        # states will be used to pass at every step to the model while training
        return prediction


def datas():
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

    print(dataX,dataY)
    print(numpy.asarray(dataX).shape)


if __name__ == '__main__':
    # datas()
    a=[[2,2,4,5,6],[2,2,4,5,6]]
    b=numpy.asarray(a)
    print(b.shape)
    # f=tf.multinomial(b,num_samples=1)
    samples = tf.random.categorical(tf.math.log([[10., 10.]]), 5)
    print(samples)