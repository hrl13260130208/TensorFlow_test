

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np


tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
list = []
texts = []
for file in os.listdir(r"C:\data\tmp2"):
    f = open(r"C:/data/tmp2/" + file, "r", encoding="utf-8").read()
    texts.append(f)
    st = tokenizer.tokenize(f)
    vocabulary_set.update(st)
    list.append(st.__len__())

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

datas=[]
labels=[]
for index,t in enumerate(texts):
    item=encoder.encode(t)
    if len(item)<1000:
        continue
    # datas.append(item)
    # labels.append(index%10)
    train_data = tf.keras.preprocessing.sequence.pad_sequences([item], padding='post', maxlen=10240)
    # print(train_data.shape)
    label_data = tf.keras.utils.to_categorical(np.array(index%10), num_classes=10)
    datas.append(train_data)
    labels.append(label_data)

train_datas=np.array(datas).reshape(113,10240)
label_datas=np.array(labels)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])



model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_datas,label_datas, epochs=10,batch_size=10)
model.save(r"C:\data\test\a")
p=model.predict(train_datas)
for i,l in enumerate(p):
    print("-------",np.argmax(l))
    print("++++++++",np.argmax(label_datas[i]))
    print("=======",np.argmax(l)==np.argmax(label_datas[i]))








