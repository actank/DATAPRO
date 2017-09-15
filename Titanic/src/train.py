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

def group_age(df, ind, col):
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
def combine_feature():
    return
def discritization():
    return
def feature_transform():
    global train, test, merge_data
    train['tt'] = 1
    test['tt'] = 0
    merge_data = pd.concat([train, test], ignore_index=True)
    merge_data = merge_data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1) 
    
    #onehot要先merge train和test
    
    

    #######################连续特征, 离散化,归一化(lr需要/gbdt不需要)################ 
    continous_data = merge_data[['Age', 'Fare']]
    #用年龄平均值补齐缺失数据
    merge_data['Age'] = merge_data['Age'].fillna(round(merge_data['Age'].mean(), 0)) 

    #测试集Fare 缺失值用平均值补齐，防止submmission行数不同不通过，不能抛弃
    merge_data['Fare'] = merge_data['Fare'].fillna(round(merge_data['Fare'].mean(), 0))

    columns = merge_data.drop(['Survived', 'tt'], axis=1).columns
    merge_data = pd.concat([merge_data['Survived'], DataFrame(normalize(merge_data.drop(['Survived', 'tt'], axis=1), norm='l1'), columns=columns)], axis=1) 
    merge_data = pd.concat([merge_data, tt], axis=1)

    #######################离散特征###############################################
    discreate_data = merge_data.drop(['Age', 'Fare', 'Survived', 'tt'], axis=1)
    #离散特征的缺失值单独成一类NTE
    embarked = merge_data['Embarked'].fillna('NTE').as_matrix()
    #merge_data['Embarked'] = LabelEncoder().fit_transform(embarked)
    embarked_label_enc = LabelEncoder().fit(embarked)

    merge_data['Embarked'] = embarked_label_enc.transform(merge_data['Embarked'].fillna('NTE'))
    sex_label_enc = LabelEncoder().fit(['male', 'female'])
    merge_data['Sex'] = sex_label_enc.transform(merge_data['Sex'].fillna('NTE'))

    #离散特征离散化
    #Pclass离散化
    tmp = None
    one_hot_enc = OneHotEncoder().fit(merge_data['Pclass'].as_matrix().reshape(-1, 1))
    tmp = one_hot_enc.transform(merge_data['Pclass'].values.reshape(-1, 1)).toarray() 
    merge_data = pd.concat([merge_data, DataFrame(np.array(tmp), columns=['Pclass#1', 'Pclass#2', 'Pclass#3'])], axis=1)

    #######################组合特征##############################################
    combine_feature = DataFrame([]) 
    #Age#Sex 年龄大的女，年龄小的男放到高维度区分，Age区间为[0-15, 16-40, 40+]
    age_sex = DataFrame([]) 
    grouped = merge_data.groupby(lambda x : group_age(merge_data, x, 'Age'))
    for name, group in grouped:
        age_sex = pd.concat([age_sex, group['Sex'].apply(lambda x : str(name) + '#' + str(x))])  
    age_sex = age_sex.rename(columns={0:"Age#Sex"})
    merge_data = pd.concat([merge_data, age_sex], axis=1)
    #PClass#Sex 仓位高的女性，更容易生还，但是一等仓的男性有可能都让给了女性，因此划分到高维，实时证明没什么卵用
    pclass_sex = DataFrame([]) 
    grouped = merge_data.groupby(['Pclass']) 
    for name, group in grouped:
        pclass_sex = pd.concat([pclass_sex, group['Sex'].apply(lambda x : str(name) + '#' + str(x))]) 
    pclass_sex = pclass_sex.rename(columns={0:"Pclass#Sex"})
    merge_data = pd.concat([merge_data, pclass_sex], axis=1)
    #Age#Sex离散化
    age_sex_label_enc = LabelEncoder().fit(merge_data['Age#Sex'])
    merge_data['Age#Sex'] = age_sex_label_enc.transform(merge_data['Age#Sex'].fillna('NTE'))
    one_hot_enc = OneHotEncoder().fit(merge_data['Age#Sex'].as_matrix().reshape(-1, 1))
    tmp = one_hot_enc.transform(merge_data['Age#Sex'].values.reshape(-1, 1)).toarray() 
    merge_data = pd.concat([merge_data, DataFrame(np.array(tmp), columns=["Age#Sex&" + str(i) for i in range(one_hot_enc.n_values_[0])])], axis=1)
    #Pclass#sex离散化
    label_enc = LabelEncoder().fit(merge_data['Pclass#Sex'])
    merge_data['Pclass#Sex'] = label_enc.transform(merge_data['Pclass#Sex'].fillna('NTE'))
    one_hot_enc = OneHotEncoder().fit(merge_data['Pclass#Sex'].as_matrix().reshape(-1, 1))
    tmp = one_hot_enc.transform(merge_data['Pclass#Sex'].values.reshape(-1, 1)).toarray() 
    tt = merge_data['tt'].copy() 
    merge_data = pd.concat([merge_data, DataFrame(np.array(tmp), columns=["Pclass#Sex&" + str(i) for i in range(one_hot_enc.n_values_[0])])], axis=1)
    
    
    #######################label###################################################
    label = merge_data[['Survivd', 'tt']]

    #######################合并连续，离散，组合feature，label#######################
    final_data = pd.concat([continous_data, discreate_data], axis=1)
    final_data = pd.concat([final_data, combine_feature], axis=1)
    final_data = pd.concat([final_data, label], axis=1)

    ###################################存数据######################################
    
    

    print(merge_data)

    #tmp = OneHotEncoder().fit_transform(data[:,col].reshape(data.shape[0], 1)).toarray()

    return

def run_train():
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
    sys.exit()  
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
    feature_transform() 
    run_train() 
    #run_gbdt_train() 
    #run_sk_gbdt_train()
    #submmit("../data/lr_predict.npy")
    submmit('../data/sk_gbdt_predict.npy')

