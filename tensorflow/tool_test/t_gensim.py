
import logging
import jieba
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import word2vec



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        file=open(self.filename,"r",encoding="utf-8")
        while(True):
            line=file.readline()

            if line:
                seg_list=jieba.cut(line)
                for str in seg_list:
                    yield str

            else:
                break

    def write(self,path):
        file=open(path,"a+",encoding="utf-8")
        for str in self:
            file.write(str+"\n")




def train(sentences):

    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)

    model.save(output)
#path = get_tmpfile("word2vec.model")




#print(model.similarity("警告","不到"))

if __name__ == '__main__':
    input="C:/File/ResultFile/input/tensorflow/organization/input"
    output = "C:/File/ResultFile/input/tensorflow/organization/out.model"
    s = word2vec.Text8Corpus(input)
    model=train(s)
    '''
    novel = "C:/File/ResultFile/input/tensorflow/novel/all.txt"
    output = "C:/File/ResultFile/input/tensorflow/novel/out.model"
    jiebaout="C:/File/ResultFile/input/tensorflow/novel/fc"
    #output="C:/File/ResultFile/input/tensorflow/word2vec/input/out.model"
    #s=MySentences("C:/File/ResultFile/input/tensorflow/word2vec/input/part-0001")
    #s=word2vec.Text8Corpus(jiebaout)
    #model=train(s)
    model=word2vec.Word2Vec.load(output)
    print(model.similar_by_vector("三重"))
'''
