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
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation
import tensorflow as tf

import sys
import os

#X_train = np.random.randn(6,2,2)
X_train = np.random.randn(6,2)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
y_train = pd.DataFrame( np.array([1, 2, 3, 4, 3, 4]) ).values
#X_test = np.random.randn(2,2,2)
X_test = np.random.randn(2,2)
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
y_test = pd.DataFrame( np.array([1, 2]) ).values

model = Sequential()
model.add(LSTM(32, 
               return_sequences=False, 
               input_dim=X_train.shape[2]))
# The shape of the last Dense layer should always correspond to y_train.shape[1]
model.add(Dense(y_train.shape[1])) 
model.add(Activation("linear"))
model.compile(loss="mean_squared_error",
              optimizer="rmsprop")

model.fit(X_train, y_train)
