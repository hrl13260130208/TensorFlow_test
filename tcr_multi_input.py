
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
            # if index > 1000:
            #     break
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
                # if index>1000:
                #     break
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

        train_data=train_data.batch(1)
        val_data=val_data.batch(1)

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
        # d=tf.split(data,10)
        out = tf.sparse_tensor_to_dense(features["out"], default_value=0)
        print(out.shape)
        return data,out

        # return text


    def get_model(self):

        if self.vocab_encoder==None:
            self.load()
        # print(self.vocab_encoder.vocab_size)
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Embedding(self.vocab_encoder.vocab_size, 64),
        #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        #     # tf.keras.layers.LSTM(1024),
        #     # tf.keras.layers.GlobalAveragePooling1D(),
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     tf.keras.layers.Dense(self.label_len, activation='softmax')
        # ])

        # i=[]
        # for index in range(10):
        #     i.append(tf.keras.Input(shape=(1024)))
        i=tf.keras.layers.Input(shape=10240)
        e=tf.keras.layers.Embedding(self.vocab_encoder.vocab_size, 64)(i)
        x=tf.keras.layers.Lambda(tf.split,arguments={"num_or_size_splits":10,"axis":1})(e)
        print(len(x))
        lstm=[]
        for input in x :
            # print(input.shape)
            lstm.append(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(input))
            # lstm.append(LSTMS(64,self.vocab_encoder.vocab_size)(input))
        #
        lstm_out=tf.keras.layers.Concatenate()(lstm)
        d1=tf.keras.layers.Dense(1024, activation='relu')(lstm_out)
        out=tf.keras.layers.Dense(self.label_len, activation='softmax')(d1)

        model = tf.keras.Model(inputs=i, outputs=out)
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


def LSTMS(units,vocab_size):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Embedding(vocab_size, 64))

    result.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units)))
    return result


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

    m=Model(subject_file=r"C:\data\tmp\t.txt",
            result_subject_dir=r"C:\data\tmp3",
            file_dir=r"C:\data\tcr_multi_input",val_data_len=100)
    # m=Model(subject_file=r"C:\data\text_classification\temp_subject_file.txt",
    #         result_subject_dir=r"C:\data\text_classification\result_subject",
    #         file_dir=r"C:\data\translate_rnn")
    # m.data_format()
    m.train()
    # tf.keras.utils.plot_model(m.get_model(),to_file="tcr_multi_input.png", show_shapes=True)
    # print(encoder)


    # for key in redis_.keys("*"):
    #     print(key)
    #     redis_.delete(key)

    # m.load_subject_dict()
    #
    # m.load_encoder()
    # labels=[]
    # features=[]
    # for line in open(r"C:\data\tmp\t.txt","r",encoding="utf-8"):
    #     item = line.replace("\n", "").split("##")
    #     label = m.subject2int_dict[item[0]]
    #     label = tf.keras.utils.to_categorical(np.array(label), num_classes=m.label_len)
    #     print(label.shape, label)
    #     text=open(os.path.join(r"C:\data\tmp3",item[1]+".txt"),"r",encoding="utf-8").read()
    #     data_feature = m.vocab_encoder.encode(text)
    #     data_feature = tf.keras.preprocessing.sequence.pad_sequences([data_feature], padding='post',
    #                                                                  maxlen=m.text_maxlen)
    #
    #     features.append(data_feature)
    #     labels.append(label)
    #
    # model=m.get_model()
    # features=np.array(features).reshape((3000,10240))
    # # features=np.split(features, 10, axis=1)
    # labels=np.array(labels)
    # # print(f.__len__(),f[0].shape,labels.shape)
    # model.fit(features,labels,batch_size=1,epochs=10)
    # tf.split(a,num_or_size_splits=10,axis=1)
    # m.get_model()
    # tf.keras.utils.plot_model(m.get_model(), to_file="tcr_multi_input.png", show_shapes=True)















