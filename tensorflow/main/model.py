import tensorflow as tf
import tflearn

class model(object):
    max_seq_len=50
    word_vec_dim=100


    def run(self):
        # 首先我们为输入的样本数据申请变量空间，如下。其中self.max_seq_len是指一个切好词的句子最多包含多少个词，self.word_vec_dim是词向量的维度，这里面shape指定了输入数据是不确定数量的样本，每个样本最多包含max_seq_len*2个词，每个词用word_vec_dim维浮点数表示。这里面用2倍的max_seq_len是因为我们训练是输入的X既要包含question句子又要包含answer句子
        input_data = tflearn.input_data(shape=[None, self.max_seq_len*2, self.word_vec_dim], dtype=tf.float32, name = "XY")
        # 然后我们将输入的所有样本数据的词序列切出前max_seq_len个，也就是question句子部分，作为编码器的输入
        encoder_inputs = tf.slice(input_data, [0, 0, 0], [-1, self.max_seq_len, self.word_vec_dim], name="enc_in")
        # 再取出后max_seq_len-1个，也就是answer句子部分，作为解码器的输入。注意，这里只取了max_seq_len-1个，是因为还要在前面拼上一组GO标识来告诉解码器我们要开始解码了，也就是下面加上go_inputs拼成最终的go_inputs
        decoder_inputs_tmp = tf.slice(input_data, [0, self.max_seq_len, 0], [-1, self.max_seq_len-1, self.word_vec_dim], name="dec_in_tmp")
        go_inputs = tf.ones_like(decoder_inputs_tmp)
        go_inputs = tf.slice(go_inputs, [0, 0, 0], [-1, 1, self.word_vec_dim])
        decoder_inputs = tf.concat(1, [go_inputs, decoder_inputs_tmp], name="dec_in")
        # 之后开始编码过程，返回的encoder_output_tensor展开成tflearn.regression回归可以识别的形如(?, 1, 200)的向量；返回的states后面传入给解码器
        (encoder_output_tensor, states) = tflearn.lstm(encoder_inputs, self.word_vec_dim, return_state=True, scope='encoder_lstm')
        encoder_output_sequence = tf.pack([encoder_output_tensor], axis=1)
        # 取出decoder_inputs的第一个词，也就是GO
        first_dec_input = tf.slice(decoder_inputs, [0, 0, 0], [-1, 1, self.word_vec_dim])
        # 将其输入到解码器中，如下，解码器的初始化状态为编码器生成的states，注意：这里的scope='decoder_lstm'是为了下面重用同一个解码器
        decoder_output_tensor = tflearn.lstm(first_dec_input, self.word_vec_dim, initial_state=states, return_seq=False, reuse=False, scope='decoder_lstm')
        # 暂时先将解码器的第一个输出存到decoder_output_sequence_list中供最后一起输出
        decoder_output_sequence_single = tf.pack([decoder_output_tensor], axis=1)
        decoder_output_sequence_list = [decoder_output_tensor]
        # 接下来我们循环max_seq_len-1次，不断取decoder_inputs的一个个词向量作为下一轮解码器输入，并将结果添加到decoder_output_sequence_list中，这里面的reuse=True, scope='decoder_lstm'说明和上面第一次解码用的是同一个lstm层
        for i in range(self.max_seq_len-1):
           next_dec_input = tf.slice(decoder_inputs, [0, i+1, 0], [-1, 1, self.word_vec_dim])
           decoder_output_tensor = tflearn.lstm(next_dec_input, self.word_vec_dim, return_seq=False, reuse=True, scope='decoder_lstm')
           decoder_output_sequence_single = tf.pack([decoder_output_tensor], axis=1)
           decoder_output_sequence_list.append(decoder_output_tensor)
        # 下面我们把编码器第一个输出和解码器所有输出拼接起来，作为tflearn.regression回归的输入
        decoder_output_sequence = tf.pack(decoder_output_sequence_list, axis=1)
        real_output_sequence = tf.concat(1, [encoder_output_sequence, decoder_output_sequence])
        net = tflearn.regression(real_output_sequence, optimizer='sgd', learning_rate=0.1, loss='mean_square')
        model = tflearn.DNN(net)