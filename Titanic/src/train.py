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
#import xgboost as xgb

import pandas as pd
from pandas import DataFrame
import numpy as np
import sys
import os
import gc


train = None
test = None
merge_data = None

def group_feature(df, ind, col):
    tmp = df[col].loc[ind]
    if tmp > 0 and tmp < 15 :
        return '1'
    elif tmp >= 15 and tmp < 45:
        return '2'
    elif tmp >=45:
        return '3'

def read_data():
    global train, test
    train = pd.read_csv("../data/train.csv", header=0)
    test = pd.read_csv("../data/test.csv", header=0) 
    return
def find_feature_columns(name, columns):
    tmp = []
    for s in columns:
        n = s.split("&")
        if len(n)>=2 and n[0] == name:
            tmp.append(s)
    return tmp
def combine_features(data1, data2, feature1, feature2):
    data1_columns = find_feature_columns(feature1, data1.columns)
    data2_columns = find_feature_columns(feature2, data2.columns)
    tmp = DataFrame([])
    for data1_column in data1_columns:
        for data2_column in data2_columns:
            cn_feature1 = data1_column.split("&")[0]
            cn_feature1_v = data1_column.split("&")[1]
            cn_feature2 = data2_column.split("&")[0]
            cn_feature2_v = data2_column.split("&")[1]
            combine_name = cn_feature1 + "&" + cn_feature2 + "#" + cn_feature1_v + "&" + cn_feature2_v
            tmp = pd.concat([tmp, (data1[data1_column] & data2[data2_column]).rename(combine_name)], axis=1)
    return tmp

#分桶离散化
def feature_discreatization(data, column, rules):
    if len(rules) < 1:
        print("discreatization rules error!") 
        return
    tmp = DataFrame([]) 
    t = DataFrame([])
    i = 0
    for rule in rules:
        t = data[(data[column] >= rule[0]) & (data[column] < rule[1])][column]
        col = "%s&%d" % (column, i)
        t = DataFrame(t).rename(columns={column: col})
        t.loc[~t[col].isnull()] = 1
        tmp = pd.concat([tmp, t])
        i = i + 1
    tmp = tmp.fillna(0).astype(np.int8)
    tmp = tmp.sort_index() 
    return tmp

def feature_transform(model_version):
    global train, test, merge_data
    ##训练集测试集标注
    train['tt'] = 1
    test['tt'] = 0


    
    #######################onehot要先合并 train和test################################
    merge_data = pd.concat([train, test], ignore_index=True)
    merge_data = merge_data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1) 


    ######################观察理解数据###############################################

    ######################采样#######################################################
  
    

    #######################连续特征, 离散化,归一化(lr需要/gbdt不需要)################ 
    continous_data = merge_data[['Age', 'Fare']].copy() 
    #用年龄平均值补齐缺失数据
    continous_data['Age'] = continous_data['Age'].fillna(round(continous_data['Age'].mean(), 0)) 
    #测试集Fare 缺失值用平均值补齐，防止submmission行数不同不通过，不能抛弃
    continous_data['Fare'] = continous_data['Fare'].fillna(round(continous_data['Fare'].mean(), 0))

    continous_columns = continous_data.columns
    if model_version == 'lr_v0.1':
        #连续特征离散化
        tmp = feature_discreatization(continous_data, 'Age', [[0, 10], [10, 16], [16, 50], [50, 100]]) 
        continous_data = pd.concat([continous_data, tmp], axis=1)
        tmp = feature_discreatization(continous_data, 'Fare', [[0, 50], [50, 1000]])
        continous_data = pd.concat([continous_data, tmp], axis=1)
        #continous_data = continous_data.drop(['Age', 'Fare'], axis=1)

    #######################离散特征###############################################
    discreate_data = merge_data.drop(['Age', 'Fare', 'Survived', 'tt'], axis=1).copy() 
    #离散特征的缺失值单独成一类NTE
    embarked = discreate_data['Embarked'].fillna('NTE').as_matrix()
    #离散特征Label编码
    #merge_data['Embarked'] = LabelEncoder().fit_transform(embarked)
    embarked_label_enc = LabelEncoder().fit(embarked)
    #embarked离散化
    discreate_data['Embarked'] = embarked_label_enc.transform(discreate_data['Embarked'].fillna('NTE'))
    one_hot_enc = OneHotEncoder().fit(discreate_data['Embarked'].as_matrix().reshape(-1, 1)) 
    tmp = one_hot_enc.transform(discreate_data['Embarked'].values.reshape(-1, 1)).toarray().astype(np.int8)
    columns = ["Embarked&%s" % i for i in one_hot_enc.active_features_]
    discreate_data = pd.concat([discreate_data, DataFrame(np.array(tmp), columns=columns)], axis=1)
    #discreate_data = discreate_data.drop(['Embarked'], axis=1)
    #sex离散化
    sex_label_enc = LabelEncoder().fit(['male', 'female'])
    discreate_data['Sex'] = sex_label_enc.transform(discreate_data['Sex'].fillna('NTE'))
    one_hot_enc = OneHotEncoder().fit(discreate_data['Sex'].as_matrix().reshape(-1, 1)) 
    tmp = one_hot_enc.transform(discreate_data['Sex'].values.reshape(-1, 1)).toarray().astype(np.int8) 
    columns = ["Sex&%s" % i for i in one_hot_enc.active_features_]
    discreate_data = pd.concat([discreate_data, DataFrame(np.array(tmp), columns=columns)], axis=1)
    #discreate_data = discreate_data.drop(['Sex'], axis=1)

    #Pclass离散化
    tmp = None
    one_hot_enc = OneHotEncoder().fit(discreate_data['Pclass'].as_matrix().reshape(-1, 1))
    tmp = one_hot_enc.transform(discreate_data['Pclass'].values.reshape(-1, 1)).toarray().astype(np.int8) 
    columns = ["Pclass&%s" % i for i in one_hot_enc.active_features_]
    discreate_data = pd.concat([discreate_data, DataFrame(np.array(tmp), columns=columns)], axis=1)
    #discreate_data = discreate_data.drop(['Pclass'], axis=1)


    #Parch离散化
    tmp = None
    one_hot_enc = OneHotEncoder().fit(discreate_data['Parch'].as_matrix().reshape(-1, 1))
    tmp = one_hot_enc.transform(discreate_data['Parch'].values.reshape(-1, 1)).toarray().astype(np.int8) 
    columns = ["Parch&%s" % i for i in one_hot_enc.active_features_]
    discreate_data = pd.concat([discreate_data, DataFrame(np.array(tmp), columns=columns)], axis=1)
    #discreate_data = discreate_data.drop(['Parch'], axis=1)

    tmp = None
    one_hot_enc = OneHotEncoder().fit(discreate_data['SibSp'].as_matrix().reshape(-1, 1))
    tmp = one_hot_enc.transform(discreate_data['SibSp'].values.reshape(-1, 1)).toarray().astype(np.int8) 
    columns = ["SibSp&%s" % i for i in one_hot_enc.active_features_]
    discreate_data = pd.concat([discreate_data, DataFrame(np.array(tmp), columns=columns)], axis=1)
    #discreate_data = discreate_data.drop(['SibSp'], axis=1)

    #######################稀疏矩阵稀疏保存######################################

    #######################组合特征##############################################
    combine_feature = DataFrame([]) 
    #Age#Sex 年龄大的女，年龄小的男放到高维度区分，Age区间为[0-15, 16-40, 40+]
    age_sex = combine_features(continous_data, discreate_data, 'Age', 'Sex')
    combine_feature = pd.concat([combine_feature, age_sex], axis=1)
    
    #Pclass&#sex没用
    
    
    #######################label###################################################
    label = merge_data[['Survived', 'tt']]

    #######################合并连续，离散，组合feature，label#######################
    final_data = pd.concat([continous_data, discreate_data], axis=1)
    final_data = pd.concat([final_data, combine_feature], axis=1)
    final_data = pd.concat([final_data, label], axis=1)

    ######################去掉原始特征#############################################

    ###################################存数据######################################
    final_data.to_csv("../data/prepare_data.csv", index=False)
     

    #tmp = OneHotEncoder().fit_transform(data[:,col].reshape(data.shape[0], 1)).toarray()

    return

def run_train():
    gc.enable() 
    gc.collect() 
    gc.disable()
    merge_data = pd.read_csv("../data/prepare_data.csv")

    tr = merge_data.loc[merge_data['tt'] == 1]
    tr = tr.drop(['tt'], axis=1)
    te = merge_data.loc[merge_data['tt'] == 0]
    te = te.drop(['tt', 'Survived'], axis=1)
    
    model = linear_model.LogisticRegression(
            penalty='l1',
            C=1e5,
            tol=1e-8,
            verbose=False)
    Y = tr['Survived']
    X = tr.drop(['Survived'], axis=1)


    #5折交叉验证
    X = X.as_matrix()
    Y = Y.as_matrix()
    kf = KFold(n_splits=6)
    kf.get_n_splits(X)
    for tr_index, te_index in kf.split(X):
        X_train, X_test = X[tr_index], X[te_index]
        Y_train, Y_test = Y[tr_index], Y[te_index]
        model.fit(X_train, Y_train)
        tr_pr = model.predict(X_train)
        te_pr = model.predict(X_test)
        train_acc = metrics.accuracy_score(tr_pr, Y_train)
        test_acc = metrics.accuracy_score(te_pr, Y_test) 
        print("train_acc: %f test_acc:%f" % (train_acc, test_acc)) 
    #交叉验证结束

    model.fit(X, Y)
    train_predict_y = model.predict(X)
    test_predict_y = model.predict(te)
    #保存predict结果
    np.save("../data/lr_predict", test_predict_y)
    print("\t")
    print("train_acc:%f" % metrics.accuracy_score(train_predict_y, tr['Survived']))
    #print("test_acc:%f" % metrics.accuracy_score(test_predict_y, test['Survived']))

    joblib.dump(model, '../data/lr.model') 
    return
def run_sk_gbdt_train():
    tr = merge_data.loc[merge_data['tt'] == 1]
    tr = tr.drop(['tt'], axis=1)
    te = merge_data.loc[merge_data['tt'] == 0]
    te = te.drop(['tt', 'Survived'], axis=1)

    Y = tr['Survived']
    X = tr.drop(['Survived'], axis=1)

    params = {'n_estimators': 500, 'max_depth': 3, 'subsample': 0.5,
            'learning_rate': 0.01, 'max_features' : 0.8, 'min_samples_split': 2, 'min_samples_leaf': 2, 'min_weight_fraction_leaf': 0.05, 'random_state': 3, 'verbose': False}

    model = ensemble.GradientBoostingClassifier(**params)
    #model.fit(X, Y)
    #param_grid = {'n_estimators':[100+100 * i for i in range(3)], 'max_depth':[3,6], 'learning_rate':[float("%1f"%(0.01 + 0.01*i)) for i in range(5)], 'verbose':[1]}
    #model = GridSearchCV(ensemble.GradientBoostingClassifier(), cv=5, param_grid=param_grid, n_jobs=20)


    #5折交叉验证
    X = X.as_matrix()
    Y = Y.as_matrix()
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    for tr_index, te_index in kf.split(X):
        X_train, X_test = X[tr_index], X[te_index]
        Y_train, Y_test = Y[tr_index], Y[te_index]
        model.fit(X_train, Y_train)
        tr_pr = model.predict(X_train)
        te_pr = model.predict(X_test)
        train_acc = metrics.accuracy_score(tr_pr, Y_train)
        test_acc = metrics.accuracy_score(te_pr, Y_test) 
        print("train_acc: %f test_acc:%f" % (train_acc, test_acc)) 
    model.fit(X, Y)

    train_predict_y = model.predict(X)
    test_predict_y = model.predict(te)
    #保存predict结果
    np.save("../data/sk_gbdt_predict", test_predict_y)
    print("\t")
    print("final train_acc:%f" % metrics.accuracy_score(train_predict_y, tr['Survived']))
    #print("final test_acc:%f" % metrics.accuracy_score(test_predict_y, ))

    return
def run_gbdt_train():
    tr = merge_data.loc[merge_data['tt'] == 1]
    tr = tr.drop(['tt'], axis=1)
    te = merge_data.loc[merge_data['tt'] == 0]
    te = te.drop(['tt', 'Survived'], axis=1)

    param = {'bst:max_depth':2, 
            'bst:eta':1, 
            'silent':1, 
            'objective':'binary:logistic',
            'eval_metric': 'auc',
            'eval_metric' : 'mae',
            'nthread' : 4
            }
    Y = train['Survived']
    X = train.drop(['Survived'], axis=1)
    num_round = 4
    watchlist  = [(test,'eval') , (train,'train')]
    bst = xgb.train( param, train, num_round, watchlist)
    bst.dump_model('../data/gbdt.model','featmap.txt')
    train_predict_y = bst.predict(X)
    test_predict_y = bst.predict(test)
    sys.exit() 

    return
def submmit(predict):
    test_data = pd.read_csv('../data/test.csv', header=0)
    predict = np.load(predict)
    with open("../data/submmission.csv", "w") as f:
        f.write('PassengerId,Survived\n')
        for i in range(len(predict)):
            f.write("%d,%d\n" % (test_data['PassengerId'].iloc[i], predict[i])) 
    return
def submit_gbdt():
    bst = xgb.Booster({'nthread':4}) #init model
    bst.load_model("model.bin") # load data
    return

if __name__ == "__main__":
    read_data() 
    feature_transform('lr_v0.1') 
    run_train() 
    #run_gbdt_train() 
    #run_sk_gbdt_train()
    #submmit("../data/lr_predict.npy")
    submmit('../data/sk_gbdt_predict.npy')

