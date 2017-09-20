#coding:utf-8
import gensim
from gensim import models
from gensim import corpora
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords  
from nltk.stem.porter import PorterStemmer   
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import sklearn
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation
import tensorflow as tf

import sys
import os

dictionary = None
english_stopwords = stopwords.words('english')
st = PorterStemmer()     
filt_stop_words = ['/','<','>',',','.',':',';','?','!','(',')','[',']','@','&','#','%','$','{','}','--','-', '``', '\\']
extra_stop_words = ["'s", "''", "'ve", "n't"]



class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0:2])

        yp = []
        for i in xrange(0, len(y_pred)):
            yp.append(y_pred[i][0])
        yt = []
        for x in self.validation_data[2]:
            yt.append(x[0])
        
        auc = roc_auc_score(yt, yp)
        self.aucs.append(auc)
        print('val-loss',logs.get('loss'), ' val-auc: ',auc)
        print('\n')
        
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


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

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
        return
    def __iter__(self):
        for f in self.filename:
            for line in open(f):
                ll = line.strip().split("\t")
                if ll[0] == "id":
                    continue
                if len(ll) == 2:
                    yield norm_text(ll[1])
                elif len(ll) == 3:
                    yield norm_text(ll[2])


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

def get_word2vec_vector():
    need_train = False
    if need_train:
        sentences = MySentences(['../data/labeledTrainData.tsv', '../data/testData.tsv'])
        model = models.Word2Vec(sentences, min_count=3, workers=3)
        model.save("../data/w2v.model")

    model = models.Word2Vec.load("../data/w2v.model")


    #转化训练集的vector
    X = []
    Y = []
    with open("../data/labeledTrainData.tsv") as f:
        for line in f:
            ll = line.strip().split("\t")
            if (ll[0] == "id"):
                continue

            feature_vec = np.zeros(() ,dtype="float32")
            doc = norm_text(ll[2])
            nwords = 0
            for word in doc:
                if word not in model.wv:
                    continue
                nwords = nwords + 1
                feature_vec = np.add(feature_vec,model.wv[word])

            feature_vec = np.divide(feature_vec, nwords)
            X.append([ (i, feature_vec[i]) for i in range(len(feature_vec))])  
            Y.append(ll[1])
    corpora.SvmLightCorpus.serialize(fname='../data/w2v_train_vector.mm', corpus=X, labels=Y)
    X = []
    Y = []
    with open("../data/testData.tsv", "r") as f:
        for line in f:
            ll = line.strip().split("\t")
            if (ll[0] == "id"):
                continue
            doc = norm_text(ll[1])
            nwords = 0
            for word in doc:
                if word not in model.wv:
                    continue
                nwords = nwords + 1
                feature_vec = np.add(feature_vec,model.wv[word])
            feature_vec = np.divide(feature_vec, nwords)
            X.append([ (i, feature_vec[i]) for i in range(len(feature_vec))])  

    #测试集不需要Y label
    corpora.SvmLightCorpus.serialize(fname='../data/w2v_test_vector.mm', corpus=X)


    return

def train(train_vector_file, test_vector_file):

    (X_te, Y_te) = load_svmlight_file(test_vector_file)
    #trick 读稀疏格式要指定n_features数目，不然模型预测时会报错测试集features与模型features数量不等，这里test的features数目比train的多，因此先读test，之后指定train的n_features为test的特征数量
    (X, Y) = load_svmlight_file(f=train_vector_file, n_features=X_te.shape[1])

    
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

def convert_sparse_matrix_to_sparse_tensor(X) :
    coo = X.tocoo() 
    indices = np.mat([coo.row, coo.col]) .transpose() 
    return tf.SparseTensor(indices, coo.data, coo.shape) 

def train_lstm(train_vector_file, test_vector_file):

    (X_te, Y_te) = load_svmlight_file(test_vector_file)
    #trick 读稀疏格式要指定n_features数目，不然模型预测时会报错测试集features与模型features数量不等，这里test的features数目比train的多，因此先读test，之后指定train的n_features为test的特征数量
    (X, Y) = load_svmlight_file(f=train_vector_file, n_features=X_te.shape[1])
    X = X.todense()
    X_te = X_te.todense()
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=20)

    #神特么技巧，要先np.array()转换后才能reshape 
    X_train = np.array(X_train) 
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val = np.array(X_val)
    X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
    X_te = np.array(X_te) 
    X_te = np.reshape(X_te, (X_te.shape[0], 1, X_te.shape[1]))

    #embedding_layer = Embedding(input_dim=X_train.shape[1],
    #                        output_dim=X_train.shape[1])

    need_train = True
    if need_train:
        print('Build model...')

        model = Sequential()
        #model.add(embedding_layer)
        model.add(keras.layers.recurrent.LSTM(units=100, 
                                            input_shape=(None, X_train.shape[2]),
                                            #input_dim=X_train.shape[2],
                                            return_sequences=False,
                                            activation='relu', 
                                            use_bias=True, 
                                            kernel_initializer='glorot_uniform', 
                                            recurrent_initializer='orthogonal', 
                                            bias_initializer='zeros', 
                                            unit_forget_bias=True, 
                                            kernel_regularizer=None, 
                                            recurrent_regularizer=None, 
                                            bias_regularizer=None, 
                                            activity_regularizer=None, 
                                            kernel_constraint=None, 
                                            recurrent_constraint=None, 
                                            bias_constraint=None, 
                                            dropout=0.2, 
                                            recurrent_dropout=0.2)) 
        #model.add(LSTM(100, input_dim=128)) 
        '''
        #双层lstm 需要更改第一层return_sequences=True，然效果并不好
        model.add(keras.layers.recurrent.LSTM(units=100, 
                                            input_shape=(None, 100),
                                            #input_dim=X_train.shape[2],
                                            return_sequences=False,
                                            activation='relu', 
                                            use_bias=True, 
                                            kernel_initializer='glorot_uniform', 
                                            recurrent_initializer='orthogonal', 
                                            bias_initializer='zeros', 
                                            unit_forget_bias=True, 
                                            kernel_regularizer=None, 
                                            recurrent_regularizer=None, 
                                            bias_regularizer=None, 
                                            activity_regularizer=None, 
                                            kernel_constraint=None, 
                                            recurrent_constraint=None, 
                                            bias_constraint=None, 
                                            dropout=0.2, 
                                            recurrent_dropout=0.2)) 
        '''

        model.add(Dense(128, input_dim=100))
        model.add(Activation('sigmoid'))
        model.add(Dense(2, input_dim=100, activation='softmax'))
        #model.layers[1].trainable=False
        histories = Histories()
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['mse'])
        model.summary() 
        
        print('Train...')
        model.fit(X_train, Y_train, batch_size=50, epochs=30,
                  validation_data=(X_train, Y_train))
        score, acc = model.evaluate(X_val, Y_val,
                                    batch_size=50)
        print('Test score:', score)
        print('Test accuracy:', acc)
        print('Train auc:', roc_auc_score(Y_train, model.predict(X_train)[:, 1])) 
        print('Val auc:', roc_auc_score(Y_val, model.predict(X_val)[:, 1]))
    else:
        model = keras.models.load_model("../data/lstm.model") 
    #test_predict_y = model.predict(X_te)[:, 1]

    test_predict_y = model.predict_classes(X_te)

    if need_train:
        model.save("../data/lstm.model")
    np.save("../data/lstm_predict", test_predict_y)

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
    #get_word2vec_vector() 
    #train("../data/w2v_train_vector.mm", "../data/w2v_test_vector.mm") 
    #train_lstm("../data/train_vector.mm", "../data/test_vector.mm")
    train_lstm("../data/w2v_train_vector.mm", "../data/w2v_test_vector.mm")
    #submission("../data/lr_predict.npy")
    submission("../data/lstm_predict.npy")
