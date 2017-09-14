#coding:utf-8
import sys
import os
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import svm
import scipy
from scipy.sparse import csr_matrix, hstack
from scipy import stats

from sklearn import linear_model
from sklearn import ensemble
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
import numpy as np
import xgboost as xgb
import pandas as pd
import gc



input_train_data = pd.DataFrame()
input_test_data = pd.DataFrame()
train_data = np.array([])
test_data = np.array([])
select_switch = True
feature_select_clf = None


def read_train():
    global input_train_data
    input_train_data = pd.read_csv("../../dataset/adult/adult.data", 
            names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','label'])
    return
def read_test():
    global input_test_data
    input_test_data = pd.read_csv("../../dataset/adult/adult.data", 
            names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','label'])

    return

def feature_extract(source):
    #global input_train_data,input_test_data
    global train_data, test_data,feature_select_clf
    
    if source == "train":
        data = input_train_data.as_matrix()
    else:
        data = input_test_data.as_matrix()

    #去空格
    data = np.char.strip(data.astype(np.str))

    combine_feature = np.empty((data.shape[0], 1), dtype=np.int8)
    ttmp = np.empty((data.shape[0], 1), dtype=np.int8)
    #+0.002education_native-country
    combine_feature = np.core.defchararray.add(data[:,3].astype(np.str), "#")
    combine_feature = np.core.defchararray.add(combine_feature, data[:,13])
    combine_feature = combine_feature.reshape((combine_feature.shape[0], 1))
    #+0.3age_occupation 
    ttmp = np.core.defchararray.add(data[:,0].astype(np.str), "#")
    combine_feature = np.column_stack((combine_feature, np.core.defchararray.add(ttmp, data[:,12]).reshape((combine_feature.shape[0], 1))))
    #-race_sex
    #ttmp = np.core.defchararray.add(data[:,8].astype(np.str), "#")
    #combine_feature = np.column_stack((combine_feature, np.core.defchararray.add(ttmp, data[:,9]).reshape((combine_feature.shape[0], 1))))

    #+0.001race_occupation
    ttmp = np.core.defchararray.add(data[:,6].astype(np.str), "#")
    combine_feature = np.column_stack((combine_feature, np.core.defchararray.add(ttmp, data[:,8]).reshape((combine_feature.shape[0], 1))))

    #+0.0005race_native-country
    #ttmp = np.core.defchararray.add(data[:,8].astype(np.str), "#")
    #combine_feature = np.column_stack((combine_feature, np.core.defchararray.add(ttmp, data[:,13]).reshape((combine_feature.shape[0], 1))))

    #=race_native-country_occupation
    #ttmp = np.core.defchararray.add(data[:,6].astype(np.str), "#")
    #ttmp = np.core.defchararray.add(data[:,8].astype(np.str), "#")
    #combine_feature = np.column_stack((combine_feature, np.core.defchararray.add(ttmp, data[:,13]).reshape((combine_feature.shape[0], 1))))
    
    #+0.001workclass_country
    ttmp = np.core.defchararray.add(data[:,1].astype(np.str), "#")
    combine_feature = np.column_stack((combine_feature, np.core.defchararray.add(ttmp, data[:,13]).reshape((combine_feature.shape[0], 1))))
    #+0.006education_hours-per-week
    ttmp = np.core.defchararray.add(data[:,3].astype(np.str), "#")
    combine_feature = np.column_stack((combine_feature, np.core.defchararray.add(ttmp, data[:,12]).reshape((combine_feature.shape[0], 1))))
    #+0.013age_education
    ttmp = np.core.defchararray.add(data[:,0].astype(np.str), "#")
    combine_feature = np.column_stack((combine_feature, np.core.defchararray.add(ttmp, data[:,3]).reshape((combine_feature.shape[0], 1))))

    #+0.01age_workclass
    ttmp = np.core.defchararray.add(data[:,0].astype(np.str), "#")
    combine_feature = np.column_stack((combine_feature, np.core.defchararray.add(ttmp, data[:,1]).reshape((combine_feature.shape[0], 1))))
    #+0.002workclass_education-num
    ttmp = np.core.defchararray.add(data[:,1].astype(np.str), "#")
    combine_feature = np.column_stack((combine_feature, np.core.defchararray.add(ttmp, data[:,4]).reshape((combine_feature.shape[0], 1))))
    #+0.004native-country_hours-per-week
    ttmp = np.core.defchararray.add(data[:,13].astype(np.str), "#")
    combine_feature = np.column_stack((combine_feature, np.core.defchararray.add(ttmp, data[:,12]).reshape((combine_feature.shape[0], 1))))
    #occupation_native-country
    ttmp = np.core.defchararray.add(data[:,6].astype(np.str), "#")
    combine_feature = np.column_stack((combine_feature, np.core.defchararray.add(ttmp, data[:,13]).reshape((combine_feature.shape[0], 1))))
    #+0.01capital-loss/(capital-loss+apital-gain)
    ttmp = data[:, 11].astype(np.int64) / (data[:, 11].astype(np.int64) + data[:, 10].astype(np.int64) + 0.01)
    combine_feature = np.column_stack((combine_feature, ttmp.reshape((combine_feature.shape[0], 1))))


    








    #组合特征单独处理
    t = csr_matrix((combine_feature.shape[1], 1)) 
    combine_feature_result = None
    for col in range(combine_feature.shape[1]):
        combine_feature[:,col] = LabelEncoder().fit_transform(combine_feature[:,col])
        #tmp = OneHotEncoder().fit_transform(combine_feature[:,col].reshape(combine_feature.shape[0], 1)).toarray()

        t = OneHotEncoder(sparse=True).fit_transform(combine_feature[:,col].reshape(combine_feature.shape[0], 1))
        if combine_feature_result == None:
            combine_feature_result = t
        else:
            combine_feature_result = hstack([combine_feature_result, t])
        #combine_feature_result = np.column_stack((combine_feature_result, tmp))


    #类别特征
    category_feature = np.empty((data.shape[0], 1), dtype=np.bool)

    tmp = np.empty((combine_feature.shape[0], 1), dtype=np.bool)
    label_col = data.shape[1] - 1
    category_col = [1,3,5,6,7,8,9,13,14]
    for col in range(data.shape[1]):
        if col in category_col:
            data[:,col] = LabelEncoder().fit_transform(data[:,col])
            tmp = OneHotEncoder().fit_transform(data[:,col].reshape(data.shape[0], 1)).toarray()
            if col != label_col:
                category_feature = np.column_stack((category_feature, tmp))
        else:
            data[:,col] = data[:,col].astype(dtype=np.int64)
    #tmp = np.array(data[:,4], dtype=np.int)
    label = data[:,label_col].copy()

    #连续特征归一化
    #删除原始类别特征
    data = np.delete(data, category_col, 1)
    data = np.delete(data, [label_col], 1)
    data = normalize(data[:,:data.shape[1]-1], norm='l1')


    #基于模型的特征选择
    if select_switch == True and feature_select_clf == None: 
        #基于pvalue的特征评估
        print(scipy.stats.pearsonr(data.astype(np.float)[:,0], label.astype(np.float)))
        print(scipy.stats.pearsonr(data.astype(np.float)[:,1], label.astype(np.float)))
        print(scipy.stats.pearsonr(data.astype(np.float)[:,2], label.astype(np.float)))
        print(scipy.stats.pearsonr(data.astype(np.float)[:,3], label.astype(np.float)))
        sys.exit()
        feature_select_clf = ExtraTreesClassifier()
        feature_select_clf = feature_select_clf.fit(data, label)
        print(data.shape)
        print(feature_select_clf.feature_importances_)
        model = SelectFromModel(feature_select_clf, prefit=True)
        data = model.transform(data)
    elif select_switch == True:
        model = SelectFromModel(feature_select_clf, prefit=True)
        data = model.transform(data)


    #合并所有特征data, combine_feature_result, l
    #data = np.column_stack((data, category_feature.astype(np.int8), label))
    #合并矩阵，存储稀疏矩阵
    data = np.column_stack((data, category_feature.astype(np.int8)))
    l = csr_matrix(label.astype(np.int8).reshape(label.shape[0], 1))
    m = csr_matrix(data)
    m = hstack((m, combine_feature_result))

    
    scipy.sparse.save_npz('./feature_matrix_%s_X.npz' % (source), m)
    scipy.sparse.save_npz('./feature_matrix_%s_Y.npz' % (source), l)

    gc.enable()
    gc.collect()
    gc.disable()

    if source == "train":
        train_data = data.copy()
    else:
        test_data = data.copy()
    return

def train_lr_l1():
    model = linear_model.LogisticRegression(
            penalty='l1',
            C=1e5,
            verbose=True)
    #Y = np.array(train_data[:,train_data.shape[1]-1], dtype=np.int)
    #X = np.array(train_data[:,:train_data.shape[1]-1], dtype=np.float)
    X_train_sparse = scipy.sparse.load_npz('./feature_matrix_train_X.npz')
    Y_train_sparse = scipy.sparse.load_npz('./feature_matrix_train_Y.npz')
    Y = Y_train_sparse.toarray()
    print(X_train_sparse.shape)
    model.fit(X_train_sparse, Y)
    #X_test = np.array(test_data[:,:test_data.shape[1]-1], dtype=np.float)
    #Y_test = np.array(test_data[:,test_data.shape[1]-1], dtype=np.int)
    X_test_sparse = scipy.sparse.load_npz('./feature_matrix_test_X.npz')
    Y_test_sparse = scipy.sparse.load_npz('./feature_matrix_test_Y.npz')
    Y_pred = model.predict(X_test_sparse)
    Y_test = Y_test_sparse.toarray()
    print("\t")
    print("train_acc:%f" % metrics.accuracy_score(model.predict(X_train_sparse), Y))
    print("test_acc:%f" % metrics.accuracy_score(Y_test, Y_pred))

    '''
    sum = 0
    count = 0
    for i in range(X_test.shape[0]):
        if X_test[i][X_test.shape[1] - 1] != Y_pred[i]:
            count += 1
        sum += 1
    '''
    print("auc:%f" % metrics.roc_auc_score(Y_pred, Y_test))
    
    return

def train_lr_l2():
    return
def train_svm():
    model = svm.LinearSVC()
    Y = np.array(train_data[:,train_data.shape[1]-1], dtype=np.int)
    X = np.array(train_data[:,:train_data.shape[1]-1], dtype=np.float)
    model.fit(X, Y)
    Y_test = np.array(test_data[:,test_data.shape[1]-1], dtype=np.int)
    X_test = np.array(test_data[:,:test_data.shape[1]-1], dtype=np.float)
    Y_pred = model.predict(X_test)
    print(model.score(X, Y))
    print(metrics.roc_auc_score(Y_pred, Y_test))

    return

def train_rf():
    return

def train_gbdt():
    tr_data = input_train_data.as_matrix()
    te_data = input_test_data.as_matrix()

    for col in range(tr_data.shape[1]):
        if col in [1,3,5,6,7,8,9,13,14]:
            tr_data[:,col] = LabelEncoder().fit_transform(tr_data[:,col])
        else:
            tr_data[:,col] = tr_data[:,col].astype(dtype=np.int64)
    for col in range(te_data.shape[1]):
        if col in [1,3,5,6,7,8,9,13,14]:
            te_data[:,col] = LabelEncoder().fit_transform(te_data[:,col])
        else:
            te_data[:,col] = te_data[:,col].astype(dtype=np.int64)
    params = {'n_estimators': 100, 'max_depth': 6, 'subsample': 0.5,
          'learning_rate': 0.09, 'min_samples_leaf': 1, 'random_state': 3, 'verbose': 0}  

    X = tr_data[:, 0:13]
    Y = np.array(tr_data[:, 14], dtype=np.int)
    X_test = te_data[:, 0:13]
    Y_test = np.array(tr_data[:, 14], dtype=np.int)
    clf = ensemble.GradientBoostingClassifier(**params)  
    #param_grid = {'n_estimators':[100+100 * i for i in range(3)], 'max_depth':[3,6], 'learning_rate':[float("%1f"%(0.01 + 0.01*i)) for i in range(5)], 'verbose':[1]}
    #clf = GridSearchCV(ensemble.GradientBoostingClassifier(), cv=5, param_grid=param_grid, n_jobs=20)
    clf.fit(X, Y)
    print(clf.score(X, Y))
    print(metrics.roc_auc_score(clf.predict(X_test), Y_test))
    Y_pred = clf.predict(X_test)
    #print(clf.best_estimator_)
    

    return

def train_xgboost():
    return

def feature_select():
    return

def train():
    read_train()
    read_test()

    global train_data, test_data
    feature_extract("train")
    feature_extract("test")

    train_lr_l1()
    #train_gbdt()
    #train_svm()
    
    return


def test():
    read_test()
    return

if __name__ == "__main__":
    train()
    test()
