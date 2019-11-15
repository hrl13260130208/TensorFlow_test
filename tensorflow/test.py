import tensorflow as tf
from nltk.corpus import PlaintextCorpusReader

import nltk
from nltk.corpus import brown
import matplotlib.pyplot as plt

import pandas as pd

import jieba.posseg as pseg

data = {"name":['google','baidu','yahoo'],"marks":[100,200,300],"price":[1,2,3]}
f=pd.DataFrame(data,index=["s","d","f"])
print(f)

'''
words = pseg.cut("广凌郡城西城的景楼大街上，行人如织，繁华热闹。")
for word, flag in words:
    print('%s %s' % (word, flag))


# 循环10次，从cfdist中取当前单词最大概率的连词,并打印出来
def generate_model(cfdist, word, num=10):
    for i in range(num):
        print(word)
        word = cfdist[word].max()


# 加载语料库
text = nltk.corpus.genesis.words('english-kjv.txt')

# 生成双连词
bigrams = nltk.bigrams(text)

# 生成条件频率分布
cfd = nltk.ConditionalFreqDist(bigrams)


# 以the开头，生成随机串
generate_model(cfd, 'the')

# 链表推导式，genre是brown语料库里的所有类别列表，word是这个类别中的词汇列表
# (genre, word)就是类别加词汇对
genre_word = [(genre, word)
              for genre in brown.categories()
              for word in brown.words(categories=genre)
              ]

# 创建条件频率分布
cfd = nltk.ConditionalFreqDist(genre_word)

# 指定条件和样本作图
cfd.plot(conditions=['news', 'adventure'],
         samples=[u'stock', u'sunbonnet', u'Elevated', u'narcotic', u'four', u'woods', u'railing', u'Until',
                  u'aggression', u'marching', u'looking', u'eligible', u'electricity', u'$25-a-plate', u'consulate',
                  u'Casey', u'all-county', u'Belgians', u'Western', u'1959-60', u'Duhagon', u'sinking', u'1,119',
                  u'co-operation', u'Famed', u'regional', u'Charitable', u'appropriation', u'yellow', u'uncertain',
                  u'Heights', u'bringing', u'prize', u'Loen', u'Publique', u'wooden', u'Loeb', u'963', u'specialties',
                  u'Sands', u'succession', u'Paul', u'Phyfe'])

path = 'C:/File/ResultFile/input/novel'
wordlists = PlaintextCorpusReader(path, '.*')
print(wordlists.raw('all.txt'))

'''