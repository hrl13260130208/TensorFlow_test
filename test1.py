import logging
import os
import tensorflow as tf
import numpy as np
import time
tf.enable_eager_execution()
import tensorflow_datasets as tfds

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

def run(path):
    time.sleep(10)
    with open(path, "r", encoding="utf-8") as f:
        text = ""
        for i in f.readlines():
            text += i + " "

        text_set = set(text.lower().split(" "))
        return text_set




def token_test():
    string="Create a list of partitioned variables according to the given slicing."
    f=open(r"C:\data\tmp2\ADA046646.txt","r",encoding="utf-8").read()
    tokenizer = tfds.features.text.Tokenizer()

    vocabulary_set = set()
    list=[]
    texts=[]
    for file in os.listdir(r"C:\data\tmp2"):
        f = open(r"C:/data/tmp2/"+file, "r", encoding="utf-8").read()
        st=tokenizer.tokenize(f)
        vocabulary_set.update(st)
        list.append(st.__len__())
        # print(st)

    list=sorted(list)
    print(list[int(list.__len__()/2)],list)
    # print(vocabulary_set.__len__())
    # encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    # list=encoder.encode(string)
    # print(type(list))
    # print(encoder.encode(st))
    # print(vocabulary_set)


def select_data():
    writer = open(r"C:\temp\subject_file_1000.txt", "w+", encoding="utf-8")
    dir=r"C:\temp\classification"
    temp={}
    with open(r"C:\Users\zhaozhijie.CNPIEC\Desktop\new_temp_subject_file.txt",encoding="utf-8") as f:
        for line in f.readlines():
            item=line.split("##")
            if item[0] in temp.keys():
                temp[item[0]].append(line)
            else:
                temp[item[0]]=[line]
        for i in range(1000):
            for key in temp.keys():
                if len(temp[key])>1000:
                    writer.write(temp[key][i])
            # writer1 = open(os.path.join(dir,key+".txt"), "w+", encoding="utf-8")
            # for index,line in enumerate(temp[key]):
            #     writer1.write(line)





if __name__ == '__main__':
    select_data()
    # token_test()
    # x = [[1, 2, 3],
    #      [1, 2, 3]]
    #
    # xx = tf.cast(x, tf.float32)
    #
    # mean_all = tf.reduce_mean(xx, keep_dims=False)
    # mean_0 = tf.reduce_mean(xx, axis=0, keep_dims=False)
    # mean_1 = tf.reduce_mean(xx, axis=1, keep_dims=False)
    # print(mean_all,mean_0,mean_1)
    # Nonparallel code
    # data=[
    #     r"C:\data\text_classification\result_subject\DE200615017155.txt",
    #     r"C:\data\text_classification\result_subject\N8625307.txt",
    #     r"C:\data\text_classification\result_subject\N140011168.txt",
    #     r"C:\data\text_classification\result_subject\DE200615017183.txt",
    #     r"C:\data\text_classification\result_subject\N150009463.txt"
    #       ]
    # s=time.time()
    # results = map(run, data)
    #
    # for r in results:
    #     print(r)
    # e = time.time()
    #
    # print("时间：", e - s)
    #
    # s = time.time()
    # # Parallel implementation
    # with ProcessPoolExecutor() as pool:
    # # with ThreadPoolExecutor(128) as pool:
    #
    #     results = pool.map(run, data)
    #     for r in results:
    #         print(r)
    #
    # e = time.time()
    #
    # print("时间：", e - s)