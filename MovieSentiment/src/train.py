#coding:utf-8
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords  
from nltk.stem.porter import PorterStemmer   

import sys
import os

dictionary = None
english_stopwords = stopwords.words('english')
st = PorterStemmer()     
filt_stop_words = ['/','<','>',',','.',':',';','?','!','(',')','[',']','@','&','#','%','$','{','}','--','-', '``', '\\']
extra_stop_words = ["'s", "''", "'ve", "n't"]


class MyCorpus(object):
    def __init__(self):
        self.__dictionary = corpora.Dictionary.load("../data/labeled_train_data_dict.dict")
        return
    def __iter__(self):
        for line in open('../data/labeledTrainData.tsv'):
            ll = line.split("\t")
            if ll[0] == 'id':
                continue
            corp = norm_text(ll[2])
            yield self.__dictionary.doc2bow(corp)

def norm_text(corp):
    corp = corp.lower().strip("\"").lstrip("\"").rstrip("\"")
    corp = corp.replace('.', ' . ') \
                .replace('<br />', ' ') \
                .replace("\"", " \" ") \
                .replace(",", " , ") \
                .replace("\"", " \" ") \
                .replace("(", " ( ") \
                .replace(")", " ) ") \
                .replace("!", " ! ") \
                .replace("?", " ? ") \
                .replace(";", " ; ") \
                .replace("\\", " \\ ") \
                .replace(":", " : ")
    texts_tokenized = [word for word in word_tokenize(corp)]
    texts_filtered_stopwords = [word for word in texts_tokenized if not word in english_stopwords and not word in filt_stop_words] 
    texts_stemmed = [st.stem(word) for word in texts_filtered_stopwords]
    texts_final = [word for word in texts_stemmed if not word in extra_stop_words]
    return texts_final

def make_dictionary():
    global dictionary
    doc = []
    english_stopwords = stopwords.words('english')
    st = PorterStemmer()     
    filt_stop_words = ['/','<','>',',','.',':',';','?','!','(',')','[',']','@','&','#','%','$','{','}','--','-', '``', '\\']
    extra_stop_words = ["'s", "''", "'ve", "n't"]
    with open("../data/labeledTrainData.tsv") as f:
        for line in f:
            ll = line.split("\t")
            id = ll[0]
            if (id == "id"):
                continue
            label = ll[1]
            #corp = ll[2].lower().strip("\"").lstrip("\"").rstrip("\"")
            corp = norm_text(ll[2])
            doc.append(corp)
    dictionary = corpora.Dictionary(doc) 
    dictionary.save("../data/labeled_train_data_dict.dict")
    return

def prepare_data():
    corpus = MyCorpus()
    for vector in corpus:
        print(vector)



    return

def train():
    return

def submission():
    return


if __name__ == "__main__":
    #make_dictionary() 
    prepare_data() 
    sys.exit() 
    train()
    sumbmmission()
