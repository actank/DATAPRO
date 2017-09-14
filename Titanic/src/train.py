#coding:utf-8
import sklearn 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn import ensemble
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib

import pandas as pd
from pandas import DataFrame
import numpy as np
import sys
import os


train = None
test = None

def read_data():
    global train, test
    train = pd.read_csv("../data/train.csv", header=0)
    test = pd.read_csv("../data/test.csv", header=0) 
    return
def feature_transform():
    global train, test
    train = train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1) 
    test = test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1) 

    #onehot要先merge train和test
    merge_data = pd.concat([train, test])
    #离散特征的缺失值单独成一类NTE
    embarked = merge_data['Embarked'].fillna('NTE').as_matrix()
    #merge_data['Embarked'] = LabelEncoder().fit_transform(embarked)
    embarked_label_enc = LabelEncoder().fit(embarked)
    train['Embarked'] = embarked_label_enc.transform(train['Embarked'].fillna('NTE'))
    test['Embarked'] = embarked_label_enc.transform(test['Embarked'].fillna('NTE'))
    #merge_data['Sex'] = LabelEncoder().fit_transform(merge_data['Sex'])
    sex_label_enc = LabelEncoder().fit(['male', 'female'])
    train['Sex'] = sex_label_enc.transform(train['Sex'].fillna('NTE'))
    test['Sex'] = sex_label_enc.transform(test['Sex'].fillna('NTE'))

    #用年龄平均值补齐缺失数据
    train['Age'] = train['Age'].fillna(round(merge_data['Age'].mean(), 0)) 
    test['Age'] = test['Age'].fillna(round(merge_data['Age'].mean(), 0)) 

    #测试集Fare 缺失值用平均值补齐，防止submmission行数不同不通过，不能抛弃
    test['Fare'] = test['Fare'].fillna(round(test['Fare'].mean(), 0))


    #其他少量缺失值，直接抛弃训练集INS数据
    train = train.dropna(axis=0, how='any')
    #tmp = OneHotEncoder().fit_transform(data[:,col].reshape(data.shape[0], 1)).toarray()

    return

def run_train():
    model = linear_model.LogisticRegression(
            penalty='l1',
            C=1e5,
            verbose=True)
    Y = train['Survived']
    X = train.drop(['Survived'], axis=1)
    model.fit(X, Y)
    train_predict_y = model.predict(X)
    test_predict_y = model.predict(test)
    print("\t")
    print("train_acc:%f" % metrics.accuracy_score(train_predict_y, train['Survived']))
    #print("test_acc:%f" % metrics.accuracy_score(test_predict_y, test['Survived']))

    joblib.dump(model, '../data/lr.model') 
    return
def submmit():
    test_data = pd.read_csv('../data/test.csv', header=0)
    model = joblib.load('../data/lr.model')
    test_predict_y = model.predict(test)
    with open("../data/submmission.csv", "w") as f:
        f.write('PassengerId,Survived\n')
        for i in range(len(test_predict_y)):
            f.write("%d,%d\n" % (test_data['PassengerId'].iloc[i], test_predict_y[i])) 
    return

if __name__ == "__main__":
    read_data() 
    feature_transform() 
    run_train() 
    submmit()

