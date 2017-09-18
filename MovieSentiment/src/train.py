#coding:utf-8
import gensim
from gensim import models
from gensim import corpora
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords  
from nltk.stem.porter import PorterStemmer   
from sklearn.model_selection import KFold
from sklearn import linear_model
import sklearn
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
import pandas as pd
import numpy as np

import sys
import os

dictionary = None
english_stopwords = stopwords.words('english')
st = PorterStemmer()     
filt_stop_words = ['/','<','>',',','.',':',';','?','!','(',')','[',']','@','&','#','%','$','{','}','--','-', '``', '\\']
extra_stop_words = ["'s", "''", "'ve", "n't"]


class MyCorpus(object):
    def __init__(self, data):
        self.__dictionary = corpora.Dictionary.load("../data/labeled_train_data_dict.dict")
        self.__data = data
        return
    def __iter__(self):
        for line in open(self.__data):
            ll = line.split("\t")
            if ll[0] == 'id':
                continue
            text = None
            if len(ll) == 2:
                text = ll[1]
            else:
                text = ll[2]
            if not text:
                print("wrong line:" + line)
                break
            corp = norm_text(text)
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
    #合并训练集测试集做词典
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
    with open("../data/unlabeledTrainData.tsv") as f:
        for line in f:
            ll = line.split("\t")
            id = ll[0]
            if (id == "id"):
                continue
            #corp = ll[2].lower().strip("\"").lstrip("\"").rstrip("\"")
            corp = norm_text(ll[1])
            doc.append(corp)

    with open("../data/testData.tsv") as f:
        for line in f:
            ll = line.split("\t")
            id = ll[0]
            if (id == "id"):
                continue
            #corp = ll[2].lower().strip("\"").lstrip("\"").rstrip("\"")
            corp = norm_text(ll[1])
            doc.append(corp)
    dictionary = corpora.Dictionary(doc) 
    dictionary.save("../data/labeled_train_data_dict.dict")
    return

def prepare_tfidf_model():

    #tfidf也要合并2个训练集和1个测试集
    #bag of words
    corp = MyCorpus("../data/labeledTrainData.tsv")
    corpus = []
    for text in corp:
        corpus.append(text)
    corp = MyCorpus("../data/testData.tsv")
    for text in corp:
        corpus.append(text)
    corp = MyCorpus("../data/unlabeledTrainData.tsv")
    for text in corp:
        corpus.append(text)
    #tfidf 合并结束
    
    #得到tfidf模型
    tfidf = models.TfidfModel(corpus)
    tfidf.save("../data/tfidf.model")
    return

def get_vector():
    tfidf_model = models.TfidfModel.load("../data/tfidf.model")
    dictionary = corpora.Dictionary.load("../data/labeled_train_data_dict.dict")
    train_data = []
    X = []
    Y = []
    with open("../data/labeledTrainData.tsv", "r") as f:
        for line in f:
            ll = line.strip().split("\t")
            if (ll[0] == "id"):
                continue
            label = ll[1]
            doc = norm_text(ll[2])
            doc2id = dictionary.doc2bow(doc)
            X.append(tfidf_model[doc2id]) 
            Y.append(label)
    corpora.SvmLightCorpus.serialize(fname='../data/train_vector.mm', corpus=X, labels=Y)
    X = []
    Y = []
    with open("../data/testData.tsv", "r") as f:
        for line in f:
            ll = line.strip().split("\t")
            if (ll[0] == "id"):
                continue
            doc = norm_text(ll[1])
            doc2id = dictionary.doc2bow(doc)
            X.append(tfidf_model[doc2id]) 

    #测试集不需要Y label
    corpora.SvmLightCorpus.serialize(fname='../data/test_vector.mm', corpus=X)

    
    return

def train():

    (X_te, Y_te) = load_svmlight_file("../data/test_vector.mm")
    #trick 读稀疏格式要指定n_features数目，不然模型预测时会报错测试集features与模型features数量不等，这里test的features数目比train的多，因此先读test，之后指定train的n_features为test的特征数量
    (X, Y) = load_svmlight_file(f="../data/train_vector.mm", n_features=X_te.shape[1])

    
    #model = svm.SVC(verbose=True)
    model = linear_model.LogisticRegression(
            penalty='l1',
            C=1e5,
            tol=1e-8,
            verbose=False)

    #10折交叉验证
    kfold_switch = False
    if kfold_switch:
        kf = KFold(n_splits=10)
        kf.get_n_splits(X)
        print("begin 10 fold cross validation")
        for tr_index, te_index in kf.split(X):
            X_train, X_test = X[tr_index], X[te_index]
            Y_train, Y_test = Y[tr_index], Y[te_index]
            model.fit(X_train, Y_train)
            tr_pr = model.predict(X_train)
            te_pr = model.predict(X_test)
            train_acc = metrics.roc_auc_score(Y_train, tr_pr)
            test_acc = metrics.roc_auc_score(Y_test, te_pr)
            print("train_acc: %f test_acc:%f" % (train_acc, test_acc))

    print("begin train model")
    model.fit(X, Y)

    train_predict_y = model.predict(X)
    test_predict_y = model.predict(X_te)
    #保存predict结果
    np.save("../data/lr_predict", test_predict_y)
    print("\t")
    print("final train_acc:%f" % metrics.roc_auc_score(Y, train_predict_y))

    return

def submission(predict):
    test_data = pd.read_csv('../data/testData.tsv', header=0, sep="\t")
    predict = np.load(predict)
    with open("../data/submission.csv", "w") as f:
        f.write('id,sentiment\n')
        for i in range(len(predict)):
            id = test_data['id'].iloc[i]
            id = id.lstrip("\"").rstrip("\"")
            f.write("%s,%d\n" % (id, predict[i]))
    return


if __name__ == "__main__":
    #make_dictionary() 
    #prepare_tfidf_model() 
    #get_vector()
    train() 
    submission("../data/lr_predict.npy")
