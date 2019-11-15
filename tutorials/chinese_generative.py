
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time

path_to_file=r"C:\data\rnn\cg\all.txt"
text = open(path_to_file,"r",encoding="utf-8").read()
# print(repr(text[:1000]))

vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
print(type(text_as_int))
# print(text_as_int[:1000])
# print(repr(char2idx["w"]))
seq_length = 100

# Create training examples / targets
chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = chunks.map(split_input_target)


# for input_example, target_example in  dataset.take(1):
#   print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
#   print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


class Model(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, units):
    super(Model, self).__init__()
    self.units = units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                     return_sequences=True,
                                     recurrent_activation='sigmoid',
                                     recurrent_initializer='glorot_uniform',
                                     stateful=True)

    self.fc = tf.keras.layers.Dense(vocab_size)

  def call(self, x):
    embedding = self.embedding(x)
    output = self.gru(embedding)
    prediction = self.fc(output)
    return prediction


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
units = 1024

model = Model(vocab_size, embedding_dim, units)


# Using adam optimizer with default arguments
optimizer = tf.train.AdamOptimizer()

# Using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)

model.build(tf.TensorShape([BATCH_SIZE, seq_length]))

model.summary()

checkpoint_dir = r"C:\data\rnn\cg\model"
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

EPOCHS = 5
# Training loop
for epoch in range(EPOCHS):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()

    for (batch, (inp, target)) in enumerate(dataset):
          with tf.GradientTape() as tape:
              # feeding the hidden state back into the model
              # This is the interesting step
              predictions = model(inp)
              loss = loss_function(target, predictions)

          grads = tape.gradient(loss, model.variables)
          optimizer.apply_gradients(zip(grads, model.variables))

          if batch % 100 == 0:
              print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,
                                                            batch,
                                                            loss))
    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
      model.save_weights(checkpoint_prefix)

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix)