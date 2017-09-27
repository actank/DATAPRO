#coding:utf-8
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn import ensemble
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM
#import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd
from pandas import DataFrame
import numpy as np
import sys
import os
import gc



def run_train():
    merge_data = pd.read_csv("../data/train_data/train.csv", header=None)
    merge_data = merge_data.fillna(0)
    X2 = merge_data.drop([0,1,27], axis=1)
    X2 = normalize(X2)
    train_data = np.concatenate((merge_data[[0,1]], X2), axis=1)
    train_data = np.concatenate((train_data, merge_data[[27]]), axis=1)

    model = linear_model.LogisticRegression(
            solver='saga',
            penalty='l1',
            C=1e5,
            tol=1e-5,
            verbose=False)
    Y = train_data[:, 27]
    X = train_data[:, :27]


    #5折交叉验证
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    for tr_index, te_index in kf.split(X):
        X_train, X_test = X[tr_index], X[te_index]
        Y_train, Y_test = Y[tr_index], Y[te_index]
        #正负样本不均衡，采样
        rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_sample(X_train, Y_train)

        model.fit(X_resampled, y_resampled)
        tr_pr = model.predict(X_train)
        te_pr = model.predict(X_test)
        #train_acc = metrics.accuracy_score(tr_pr, Y_train)
        #test_acc = metrics.accuracy_score(te_pr, Y_test)
        f1_score_train = metrics.f1_score(Y_train, tr_pr)
        f1_score_test = metrics.f1_score(Y_test, te_pr)
        print("train_f1:%f test_f1:%f" % (f1_score_train, f1_score_test))
    #交叉验证结束
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_sample(X, Y)

    model.fit(X_resampled, y_resampled)
    train_predict_y = model.predict(X)
    #test_predict_y = model.predict(te)
    #保存predict结果
    #np.save("../data/lr_predict", test_predict_y)
    print("\t")
    print("final train_f1_score:%f" % (metrics.f1_score(Y, train_predict_y)))
    #print("test_acc:%f" % metrics.accuracy_score(test_predict_y, test['Survived']))

    joblib.dump(model, '../data/lr.model')
    return

if __name__ == "__main__":
    run_train()
