
import time
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_datasets as tfds
import os
import numpy as np
import  redis
import json
from concurrent.futures import ThreadPoolExecutor


redis_ = redis.Redis(host="10.3.1.99", port=6379, db=5,decode_responses=True)
VOCAB_SET="vocab_set"
USE_REDIS=True
LOCAL_VOCAB_SET=set()

tokenizer = tfds.features.text.Tokenizer()

# encoder = tfds.features.text.TokenTextEncoder()
# encoder=tfds.features.text.TokenTextEncoder.load_from_file(r"C:\data\1\vocab_encoder")


class Encoder():
    pass


class Model():

    def __init__(self,subject_file=r"C:\data\text_classification\temp_subject_file.txt",
                    result_subject_dir=r"C:\data\text_classification\result_subject",
                    file_dir=r"C:\data\translate_rnn",
                    subject_dict_file_name="subjet_dict",
                    vocab_encoder_file_name="vocab_encoder",
                    model_name="text_model",
                    text_data_name="text_file",
                    tf_data_name="tf_file",
                    # label_data_name="label_file",
                    val_data_len=1000,
                    text_maxlen = 10240):

        self.subject_file=subject_file
        self.result_subject_dir=result_subject_dir
        self.file_dir=file_dir

        self.text_maxlen=text_maxlen

        self.subject_dict_path = os.path.join(self.file_dir,subject_dict_file_name)
        self.vocab_encoder_path = os.path.join(self.file_dir, vocab_encoder_file_name)

        self.text_data_file_path = os.path.join(self.file_dir, text_data_name)
        self.tf_data_file_path = os.path.join(self.file_dir, tf_data_name)

        self.vocab_encoder=None
        self.subject2int_dict=None
        self.int2subject_dict=None
        self.label_len=-1

        self.val_data_len=val_data_len

        self.model_path= os.path.join(self.file_dir,model_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)


    def load_encoder(self):

        if os.path.exists(self.vocab_encoder_path+".tokens"):
            self.vocab_encoder = tfds.features.text.TokenTextEncoder.load_from_file(self.vocab_encoder_path)


    def load_subject_dict(self):
        if os.path.exists(self.subject_dict_path):
            with open(self.subject_dict_path, "r+", encoding="utf-8") as f:
                lines = f.readlines()
                self.subject2int_dict = json.loads(lines[0])
                self.int2subject_dict = json.loads(lines[1])
                self.label_len=len(self.subject2int_dict.keys())


    def data_format(self):
        print("解析字典文件...")
        self.load()

        print("生成tfrecord文件...")
        path_list = []
        f_subject = open(self.subject_file, "r", encoding="utf-8")
        for index, line in enumerate(f_subject.readlines()):
            if index > 1000:
                break
            item = line.replace("\n", "").split("##")
            path_list.append((item[0], os.path.join(self.result_subject_dir, item[1] + ".txt")))

        print("读取文本文件...")

        tf_data_writer = tf.python_io.TFRecordWriter(self.tf_data_file_path)
        s = time.time()
        with ThreadPoolExecutor(256) as pool:

            results = pool.map(read_txt, path_list)

            for r in results:
                line = ""
                for w in r[1]:
                    line += w + " "
                self.write_tf_file(r[0], line, tf_data_writer)
        e = time.time()
        print("读取文件用时：", e - s)

    def write_tf_file(self,line, text, writer):
        print(line)
        label=self.subject2int_dict[line]
        label = tf.keras.utils.to_categorical(np.array(label), num_classes=self.label_len)
        print(label.shape,label)
        data_feature=self.vocab_encoder.encode(text)
        data_feature= tf.keras.preprocessing.sequence.pad_sequences([data_feature], padding='post', maxlen=self.text_maxlen)

        features = tf.train.Features(
            feature={
                "input": tf.train.Feature(int64_list=tf.train.Int64List(value=data_feature[0])),
                "out": tf.train.Feature(float_list=tf.train.FloatList(value=label))
            }
        )
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)

    def load(self):

        self.load_subject_dict()

        self.load_encoder()

        if self.subject2int_dict == None or self.vocab_encoder == None:
            print("文件不完整，重新生成...")
            f_subject = open(self.subject_file, "r", encoding="utf-8")
            subject_set = set()
            # vocab_set = set()
            path_list = []
            print("读取subject文件...")
            for index,line in enumerate(f_subject.readlines()):
                if index>1000:
                    break
                item = line.replace("\n", "").split("##")
                subject_set.add(item[0])
                path_list.append((item[0], os.path.join(self.result_subject_dir, item[1] + ".txt")))

            print("读取文本文件...")


            s=time.time()
            with ThreadPoolExecutor(256) as pool:

                results = pool.map(read_txt_to_set, path_list)
                # print(len(results))
                for r in results:
                    print(r[0])
            e=time.time()
            print("读取文件用时：",e-s)
            vocab_set = redis_.smembers(VOCAB_SET)
            self.subject2int_dict = {u: i for i, u in enumerate(subject_set)}
            self.int2subject_dict = {i: u for i, u in enumerate(subject_set)}
            self.vocab_encoder = tfds.features.text.TokenTextEncoder(vocab_set)
            self.label_len=len(self.subject2int_dict.keys())

            print("生成字典文件...")
            fs = open(self.subject_dict_path, "w+", encoding="utf-8")
            fs.write(json.dumps(self.subject2int_dict) + "\n" + json.dumps(self.int2subject_dict))
            fs.close()
            self.vocab_encoder.save_to_file(self.vocab_encoder_path)

    def train(self):

        if self.vocab_encoder==None:

            self.load_encoder()
        if self.subject2int_dict == None:
            self.load_subject_dict()


        dataset = tf.data.TFRecordDataset(self.tf_data_file_path)
        dataset=dataset.map(self.item_map)
        # for l in dataset.take(10):
        #     print(l)
        # dataset=dataset.batch(10)

        train_data = dataset.skip(self.val_data_len)
        val_data = dataset.take(self.val_data_len)

        train_data=train_data.batch(10)
        val_data=val_data.batch(10)

        train_data = train_data.repeat()
        
        latest = tf.train.latest_checkpoint(self.model_path)
        model = self.get_model()
        if latest != None:
            model.load_weights(latest)
        checkpoint_path = self.model_path + "/cp-{epoch:04d}.ckpt"

        # 创建一个保存模型权重的回调
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=5)
        model.fit(train_data, epochs=10, steps_per_epoch=30,validation_data=val_data,validation_steps=10,callbacks=[cp_callback])


    def item_map(self,text):
        features = tf.parse_single_example(text,
                                           features={'input': tf.VarLenFeature(tf.int64),
                                                     "out": tf.VarLenFeature(tf.float32)
                                                     })

        data = tf.sparse_tensor_to_dense(features["input"], default_value=0)

        out = tf.sparse_tensor_to_dense(features["out"], default_value=0)
        print(out.shape)
        return data,out

        # return text


    def get_model(self):

        if self.vocab_encoder==None:
            self.load()
        print(self.vocab_encoder.vocab_size)
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_encoder.vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(self.label_len, activation='softmax')
        ])



        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(0.001),
                      metrics=['accuracy'])
        model.summary()
        return model

    def predict(self, data):
        latest = tf.train.latest_checkpoint(self.model_path)
        model = self.get_model()
        model.load_weights(latest)
        return model.predict(data)

    def get_predict_data(self, file_path):
        f = open(file_path, "r", encoding="utf-8")
        if self.vocab_encoder==None:
            self.load_encoder()


        text= f.read().lower().replace("\n"," ")
        ints=self.vocab_encoder.encode(text)

        return tf.keras.preprocessing.sequence.pad_sequences([ints], padding='post', maxlen=self.text_maxlen)

    def predict_file(self,file_path):
        print("预测文件：",file_path)
        if self.int2subject_dict==None:
            self.load_subject_dict()

        p=self.predict(self.get_predict_data(file_path))
        int_p=np.argmax(p)
        print("预测类别：",self.int2subject_dict[int_p])

    def predict_dir(self,dir):
        for f_name in os.listdir(dir):
            self.predict_file(os.path.join(dir,f_name))


def read_txt(item,all_lower=True):
    path = item[1]
    # print("读取文件：", path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().replace("\n", " ")
        if all_lower:
            text = text.lower()
    return item[0],text

def read_txt_to_set(item,use_redis=True,all_lower=True):
    path=item[1]
    print("读取文件：",path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().replace("\n", " ")
        if all_lower:
            text=text.lower()
        spt=tokenizer.tokenize(text)

        #添加到set
        if use_redis:
            for w in spt:
                redis_.sadd(VOCAB_SET,w)

        return item[0],spt


# def read_text_map(text):
#     print(text.numpy())
#
#
#     return text
#
# def encode_map_fn(text):
#   return tf.py_function(read_text_map,inp=[text], Tout=(tf.string))
if __name__ == '__main__':

    m=Model(subject_file=r"C:\data\text_classification\temp_subject_file.txt",
            result_subject_dir=r"C:\data\text_classification\result_subject",
            file_dir=r"C:\data\1",val_data_len=100)
    # m=Model(subject_file=r"C:\data\text_classification\temp_subject_file.txt",
    #         result_subject_dir=r"C:\data\text_classification\result_subject",
    #         file_dir=r"C:\data\translate_rnn")
    m.data_format()
    m.train()
    # print(encoder)


    # for key in redis_.keys("*"):
    #     print(key)
    #     redis_.delete(key)


    # vocabulary_set = set()
    # list = []
    # texts = []
    # for file in os.listdir(r"C:\data\tmp2"):
    #     f = open(r"C:/data/tmp2/" + file, "r", encoding="utf-8").read()
    #     texts.append(f)
    #     st = tokenizer.tokenize(f)
    #     # print(st)
    #     vocabulary_set.update(st)
    #     list.append(st.__len__())
    #
    #
    # encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    #
    # datas=[]
    # labels=[]
    # for index,t in enumerate(texts):
    #     item=encoder.encode(t)
    #     if len(item)<1000:
    #         continue
    #     # datas.append(item)
    #     # labels.append(index%10)
    #     train_data = tf.keras.preprocessing.sequence.pad_sequences([item], padding='post', maxlen=10240)
    #     # print(train_data.shape)
    #     label_data = tf.keras.utils.to_categorical(np.array(index%10), num_classes=10)
    #     datas.append(train_data)
    #     labels.append(label_data)
    #
    # train_datas=np.array(datas).reshape(113,10240)
    # label_datas=np.array(labels)
    #
    # print(train_datas.shape,encoder.vocab_size)
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(encoder.vocab_size, 64),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    #     tf.keras.layers.Dense(1024, activation='relu'),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])
    #
    #
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=tf.keras.optimizers.Adam(0.001),
    #               metrics=['accuracy'])
    #
    # history = model.fit(train_datas,label_datas, epochs=10,batch_size=10)
#
# model.save(r"C:\data\test\a")
# # model.load_weights(r"C:\data\test\a")
# p=model.predict(train_datas)
# t=0
# for i,l in enumerate(p):
#     print("-------",np.argmax(l))
#     print("++++++++",np.argmax(label_datas[i]))
#     print("=======",np.argmax(l)==np.argmax(label_datas[i]))
#     if np.argmax(l)==np.argmax(label_datas[i]):
#         t+=1
#
# print(t,t/len(p))








