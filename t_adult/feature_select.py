#coding:utf-8

import scipy
from scipy import sparse, stats

X_train_sparse = scipy.sparse.load_npz('./feature_matrix_train_X.npz')
Y_train_sparse = scipy.sparse.load_npz('./feature_matrix_train_Y.npz')
print(X_train_sparse)
print(scipy.stats.pearsonr(X_train_sparse, Y_train_sparse))
