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
import xgboost as xgb
import lightgbm as lgb

import pandas as pd
from pandas import DataFrame
import numpy as np
import sys
import os


def train_xgb():
    train_data = pd.read_csv("../data/train")
    X = train_data.drop(['click'], axis=1)
    Y = train_data
    params={
        'booster':'gbtree',
        #'objective': 'multi:softmax', #多分类的问题
        'objective':'binary:logistic',
        #'num_class':10, # 类别数，与 multisoftmax 并用
        #'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth':5, # 构建树的深度，越大越容易过拟合
        'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample':0.9, # 随机采样训练样本
        'colsample_bytree':0.9, # 生成树时进行的列采样
        #'min_child_weight':3,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.02, # 如同学习率
        'seed':1000,
        'nthread':23,# cpu 线程数
        'verbose':0,
        'eval_metric': 'auc'
    }
    plst = list(params.items())
    num_rounds = 2000 # 迭代次数,树个数
    #10折交叉验证
    kfold_switch = False
    kfold = 5
    if kfold_switch:
        kf = KFold(n_splits=kfold)
        kf.get_n_splits(X)
        print("begin 5 fold cross validation")
        for tr_index, te_index in kf.split(X):
            X_train, X_test = X.iloc[tr_index], X.iloc[te_index]
            Y_train, Y_test = Y.iloc[tr_index], Y.iloc[te_index]
            xgb_train = xgb.DMatrix(X_train, label=Y_train)
            xgb_test = xgb.DMatrix(X_test, label=Y_test)
            watchlist = [(xgb_train, 'train'),(xgb_test, 'val')]
            model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)
    return

if __name__ == "__main__":
    train_xgb()
