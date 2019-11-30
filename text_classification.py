import logging
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json
import time
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import redis

redis_ = redis.Redis(host="192.168.1.20", port=6379, db=3,decode_responses=True)
VOCAB_SET="vocab_set"

class Model():

    def __init__(self,subject_file=r"C:\data\text_classification\temp_subject_file.txt",
                    result_subject_dir=r"C:\data\text_classification\result_subject",
                    file_dir=r"C:\data\text_classification",
                    subject_dict_file_name="subjet_dict",
                    vocab_dict_file_name="vocab_dict",
                    train_data_file_name="train_tfrecord_file",
                    val_data_file_name="val_tfrecord_file",
                    model_name="text_model",
                    feature_data_name="feature_file",
                    label_data_name="label_file"):

        self.subject_file=subject_file
        self.result_subject_dir=result_subject_dir
        self.file_dir=file_dir

        self.subject_dict_file_name=subject_dict_file_name
        self.vocab_dict_file_name=vocab_dict_file_name
        self.subject_dict_path = os.path.join(self.file_dir, self.subject_dict_file_name)
        self.vocab_dict_path = os.path.join(self.file_dir, self.vocab_dict_file_name)

        # self.train_data_file_name=train_data_file_name
        # self.val_data_file_name=val_data_file_name
        self.train_data_file_path = os.path.join(self.file_dir, train_data_file_name)
        self.val_data_file_path = os.path.join(self.file_dir, val_data_file_name)

        self.feature_data_file_path = os.path.join(self.file_dir, feature_data_name)
        self.label_data_file_path = os.path.join(self.file_dir, label_data_name)


        self.model_path= os.path.join(self.file_dir,model_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)


    def get_model(self):
        len=self.get_vocb_lenth()
        onehot_num=self.get_subject_lenth()
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Embedding(len, 50))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Dense(2560, activation='relu'))
        model.add(tf.keras.layers.Dense(onehot_num, activation='softmax'))

        model.summary()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def get_vocb_lenth(self):
        with open(self.vocab_dict_path, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            vocab2int_dict = json.loads(lines[0])
            return vocab2int_dict.keys().__len__()
    def get_subject_lenth(self):
        with open(self.subject_dict_path, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            subject2int_dict = json.loads(lines[0])
            return subject2int_dict.keys().__len__()




    def data_format(self):

        print("开始处理数据...")

        subject2int_dict ,int2subject_dict ,vocab2int_dict =self.get_dicts()

        print("开始生成TFRecord文件...")
        # print(vocab2int_dict)
        subject_num=subject2int_dict.keys().__len__()

        train_data_writer=tf.python_io.TFRecordWriter(self.train_data_file_path)
        val_data_writer=tf.python_io.TFRecordWriter(self.val_data_file_path)
        with open(self.label_data_file_path,"r",encoding="utf-8") as f:
            lines=f.readlines()
            texts=open(self.feature_data_file_path,"r",encoding="utf-8").readlines()
            result_list=[]

            for index,line in enumerate(lines):
                # print(index)
                if index>1000:
                    break
                if index%10>3 :
                    write_tf_file(line,texts[index],train_data_writer,vocab2int_dict,subject2int_dict,subject_num)
                else:
                    write_tf_file(line,texts[index], val_data_writer, vocab2int_dict, subject2int_dict, subject_num)

            # for line in lines[:int(lines.__len__()/10*7)]:
            #     self.write_tf_file(line,train_data_writer,vocab2int_dict,subject2int_dict,subject_num)
            # for line in lines[int(lines.__len__()/10*7):]:
            #     self.write_tf_file(line,val_data_writer,vocab2int_dict,subject2int_dict,subject_num)


    def get_dicts(self):
        print("解析字典文件...")
        subject2int_dict = None
        int2subject_dict = None
        vocab2int_dict = None
        if os.path.exists(self.subject_dict_path):
            with open(self.subject_dict_path,"r+",encoding="utf-8") as f:
                lines=f.readlines()
                subject2int_dict=json.loads(lines[0])
                int2subject_dict=json.loads(lines[1])

        if os.path.exists(self.vocab_dict_path):
            with open(self.vocab_dict_path, "r+", encoding="utf-8") as f:
                lines = f.readlines()
                vocab2int_dict = json.loads(lines[0])


        if subject2int_dict == None or vocab2int_dict == None:
            print("字典文件不完整，重新生成...")
            f_subject = open(self.subject_file, "r", encoding="utf-8")
            subject_set = set()
            # vocab_set = set()
            path_list=[]
            for line in f_subject.readlines():

                item = line.replace("\n", "").split("##")
                print("处理文件" + item[1] + "...")
                subject_set.add(item[0])
                path_list.append((item[0],os.path.join(self.result_subject_dir, item[1] + ".txt")))
                # with open(os.path.join(self.result_subject_dir, item[1] + ".txt"), "r", encoding="utf-8") as f:
                #     text = ""
                #     for i in f.readlines():
                #         text += i + " "
                #
                #     text_set = set(text.lower().split(" "))
                #     vocab_set = vocab_set | text_set

            if redis_.keys(VOCAB_SET)==None:
                print("词表为空！")
                fd=open(self.feature_data_file_path,"w+",encoding="utf-8")
                ld=open(self.label_data_file_path,"w+",encoding="utf-8")


                s = time.time()

                with ThreadPoolExecutor(128) as pool:

                    results = pool.map(read_txt_to_set, path_list)
                    # print(len(results))
                    i=0
                    for r in results:
                        i+=1
                        ld.write(r[0]+"\n")
                        fd.write(r[1].lower()+"\n")

                e = time.time()

                print("时间：", e - s)
            # with open(self.feature_data_file_path, encoding="utf-8") as f:
            #     for line in f.readlines():
            #         redis_.sadd(VOCAB_SET,line.replace("\n", " ").split(" "))
            vocab_set=redis_.smembers(VOCAB_SET)
            subject2int_dict = {u: i for i, u in enumerate(subject_set)}
            int2subject_dict = {i: u for i, u in enumerate(subject_set)}
            vocab2int_dict = {u: i for i, u in enumerate(vocab_set)}

            print("生成字典文件...")
            fs = open(self.subject_dict_path, "w+", encoding="utf-8")
            fs.write(json.dumps(subject2int_dict) + "\n" + json.dumps(int2subject_dict))
            fs.close()
            fv = open(self.vocab_dict_path, "w+", encoding="utf-8")
            fv.write(json.dumps(vocab2int_dict))
            fv.close()
        return subject2int_dict,int2subject_dict,vocab2int_dict


    def train(self):

        dataset = tf.data.TFRecordDataset(self.train_data_file_path)
        dataset = dataset.map(read_tfrecord_map)
        dataset = dataset.batch(100)
        dataset = dataset.repeat()

        val_data=tf.data.TFRecordDataset(self.val_data_file_path)
        val_data=val_data.map(read_tfrecord_map)
        val_data=val_data.batch(100)
        val_data=val_data.repeat()
        # for data in dataset.take(1):
        #     print("----------------",data)



        latest = tf.train.latest_checkpoint(self.model_path)
        model = self.get_model()
        if latest!=None:
            model.load_weights(latest)
        checkpoint_path = self.model_path+"/cp-{epoch:04d}.ckpt"

        # 创建一个保存模型权重的回调
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=5)
        histry=model.fit(dataset, epochs=10, steps_per_epoch=30)


    def predict(self,data):
        latest = tf.train.latest_checkpoint(self.model_path)
        model=self.get_model()
        model.load_weights(latest)
        return model.predict(data)

    def get_predict_data(self,file_path,vocab2int_dict):
        f=open(file_path,"r",encoding="utf-8")

        lines=f.readlines()
        int_list=[]
        for word in lines[0].lower().split(" "):
            try:
                int_list.append(vocab2int_dict[word])
            except:
                pass

        return int_list

    def predict_text(self,file):
        _,int2subject,vocab2int=self.get_dicts()
        data=self.get_predict_data(file,vocab2int)
        p_data = tf.keras.preprocessing.sequence.pad_sequences([data], padding='post', maxlen=51200)
        p=self.predict(p_data)
        print(np.argmax(p[0]),p)
        return int2subject[str(np.argmax(p))]

    def predict_dir(self,dir):
        _, int2subject, vocab2int = self.get_dicts()
        datas=[]
        for name in os.listdir(dir):

            data = self.get_predict_data(os.path.join(dir,name), vocab2int)
            datas.append(data)
        p_data = tf.keras.preprocessing.sequence.pad_sequences(datas, padding='post', maxlen=51200)
        p=self.predict(p_data)
        r=[]
        for s in p:
            print(str(np.argmax(s)))
            r.append(int2subject[str(np.argmax(s))])
        return r



def read_tfrecord_map(line):
    # tf.p
    features = tf.parse_single_example(line,
                                       features={'input': tf.VarLenFeature(tf.int64),
                                                 "out": tf.VarLenFeature(tf.float32)
                                                 })

    data = tf.sparse_tensor_to_dense(features["input"], default_value=0)
    # train_data=
    out = tf.sparse_tensor_to_dense(features["out"], default_value=0)
    return data,out


def read_txt_to_set(item):
    path=item[1]
    print(path)
    with open(path, "r", encoding="utf-8") as f:
        text = ""
        for i in f.readlines():
            text += i.replace("\n","") + " "
        for word in text.lower().split(" "):
            redis_.sadd(VOCAB_SET,word)
        return (item[0],text.replace("\n"," "))

def write_tf_file(line,text,writer,vocab2int_dict,subject2int_dict,subject_num):
    print(line)
    line=line.replace("\n","")
    items = text.lower().replace("\n"," ").split(" ")
    int_list = []
    try:
        for item in items:
            int_list.append(int(vocab2int_dict[item]))
    except:
        print("处理出错！")
        return

    sint = subject2int_dict[line]
    # print(int_list)
    train_data = tf.keras.preprocessing.sequence.pad_sequences([int_list], padding='post', maxlen=51200)
    # print(train_data.shape)
    label_data = tf.keras.utils.to_categorical(np.array(sint), num_classes=subject_num)
    print(sint,label_data)
    features = tf.train.Features(
        feature={
            "input": tf.train.Feature(int64_list=tf.train.Int64List(value=train_data.flatten())),
            "out": tf.train.Feature(float_list=tf.train.FloatList(value=label_data))
        }
    )
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)

if __name__ == '__main__':

    m=Model(subject_file=r"C:\data\text_classification\temp_subject_file.txt",
            result_subject_dir=r"C:\data\text_classification\result_subject",
            file_dir=r"C:\data\text_classification")
    m.data_format()
    m.train()
    # s=time.time()
    # # list=[]
    # # with open(r"C:\data\text_classification\feature_file", encoding="utf-8") as f:
    # #     for line in f.readlines():
    # #         list.extend(line.replace("\n", " ").split(" "))
    # list=open(r"C:\data\text_classification\feature_file", encoding="utf-8").read()
    #
    # tokenizer = tfds.features.text.Tokenizer()
    #
    #
    # st = tokenizer.tokenize(list)
    # vocabulary_set = set(st)
    # e = time.time()
    # print("时间：", e - s)
    # vocab_set = sorted(vocabulary_set)
    # dict().keys()
    # dict().values()

    # m.get_dicts()

    # print(m.predict_text(r"C:\data\text_classification\result_subject\AD1011810.txt"))
    # print(m.predict_dir(r"C:\data\test\testdata"))




