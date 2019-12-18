
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()
tf.enable_eager_execution()
import tensorflow_datasets as tfds
import random
import numpy as np
import os
import json


tokenizer = tfds.features.text.Tokenizer()

train=[]
label=[]

vocab=set()
temp=[]
for file in os.listdir(r"C:\data\tmp2"):
    path = os.path.join(r"C:\data\tmp2",file)
    f=open(path,encoding="utf-8")
    text=f.read()
    s=tokenizer.tokenize(text)
    if s.__len__()<1000:
        continue
    vocab.update(s)
    temp.append(text)

    label_i=np.zeros([10])
    for i in range(int(random.Random().random()*5)):
        label_i[int(random.Random().random()*10)]=1.0
    # print(label_i)
    label.append(label_i)
encoder=tfds.features.text.TokenTextEncoder(vocab)

for t in temp:
    i=encoder.encode(t)
    # print(i)
    train.append(i)

train= tf.keras.preprocessing.sequence.pad_sequences(train, padding='post', maxlen=10240)
label=np.array(label)
# print(train.shape)
# print(label.shape)
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:2223","localhost:2224"]
    },
    'task': {'type': 'worker', 'index': 0}
})
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(encoder.vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(10, activation='sigmoid')
        ])


    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['accuracy'])
    model.summary()
model.fit(train,label,batch_size=10,epochs=10)






