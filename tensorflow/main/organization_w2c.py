from gensim.models import word2vec
import pandas as pd



def trian():
    #input = "C:/File/ResultFile/input/tensorflow/organization/input"
    #output = "C:/File/ResultFile/input/tensorflow/organization/out.model"
    input = "C:/Users/zhaozhijie.CNPIEC/Desktop/导出.tsv"
    output = "C:/File/ResultFile/input/tensorflow/word2vec/output/out.model"
    sentences = word2vec.Text8Corpus(input)
    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=0, workers=4)
    model.save(output)

def trian_new_data():
    input = "C:/Users/zhaozhijie.CNPIEC/Desktop/导出.tsv"
    output = "C:/File/ResultFile/input/tensorflow/word2vec/output/out.model"
    sentences = word2vec.Text8Corpus(input)
    model=word2vec.Word2Vec.load(output)
    model.train(sentences,total_examples=711389,epochs=10)
    model.save(output)


def test():
    input="C:/File/data/word2vec/test_s.txt"
    output = "C:/File/ResultFile/input/tensorflow/word2vec/output/out.model"
    model = word2vec.Word2Vec.load(output)
    f_input=open(input,encoding="utf-8")
    s=[]
    for line in f_input.readlines():
        s.append( line.replace("\n",""))
    print(s)

    err=[]
    for l in  s:
        for word in l.split(" "):
            if word not in model.wv.vocab:
                print("input word %s not in dict. skip this turn" % word)
                err.append(l)
                s.remove(l)

    data={}
    for line in s:

        row_data=[]
        l_words=line.split(" ")
        for row in s:
            r_words= row.split(" ")
            score = model.n_similarity(l_words,r_words)
            row_data.append(score)
            if score>0.9:
                print(score ,"\t "+line ,"\t "+row)

        data[line]=row_data
    f=pd.DataFrame(data,index=s)
    f.to_excel("C:/File/data/word2vec/out.xlsx")


def test2():
    output = "C:/File/ResultFile/input/tensorflow/word2vec/output/out.model"
    model = word2vec.Word2Vec.load(output)
    print(model["ESALQ/USP\""])


def test3():
    output = "C:/File/ResultFile/input/tensorflow/word2vec/output/out.model"
    model = word2vec.Word2Vec.load(output)
    str1="Mechanical Engineering Faculty, Tarbiat Modares University"
    str2="Mechanical Engineering Department, Faculty of Engineering, Tarbiat Modares University"
    score = model.wv.n_similarity(str1.split(" "), str2.split(" "))
    print(score)

if __name__ == '__main__':
    output = "C:/File/data/novel/out.model"
    model = word2vec.Word2Vec.load(output)
    print(model["飞剑"])






'''
output = "C:/File/ResultFile/input/tensorflow/organization/out.model"
model=word2vec.Word2Vec.load(output)


candidates = ["Hospital, Capital Medical University and the Center of Stroke",
              "Department of Breast Surgery, Nagoya Medical Center",
              "Orthopädie für die Universität Regensburg, Klinikum Bad Abbach",
              "NSABP Foundation, Inc., Nova Tower 2, Two Allegheny Center, 12th Flr.",
              "Division of Cardiology, Department of Internal Medicine, Faculty of Medicine, Songklanagarind Hospital, Prince of Songkla University",
              "Klinikvorstand der Universitätszahnklinik Wien, Medizinische Universität Wien",
              "Department of Childhood Education, Faculty of Human Sciences, Tohoku Bunkyo University"]

text = "Neurology Department and Cerebral Vascular Diseases Research Institute (China-America Institute of Neuroscience), Xuanwu Hospital, Capital Medical University and the Center of Stroke, Beijing Institute for Brain Disorders"  # 待匹配文本
words = text.split(" ") # 分词
flag = False
word = []
for w in words:
    if w not in model.wv.vocab:
        print("input word %s not in dict. skip this turn" % w)
    else:
        word.append(w)
# 文本匹配
res = []

index = 0
for candidate in candidates:
    for c in candidate.split(" "):
        if c not in model.wv.vocab:
            print("candidate word %s not in dict. skip this turn" % c)
            flag = True
    if flag:
        break
    score = model.n_similarity(word, candidate.split(" "))
    resultInfo = {'id': index, "score": score, "text": candidate}
    res.append(resultInfo)
    print(resultInfo)
    index += 1

'''







