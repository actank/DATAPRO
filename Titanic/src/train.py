#coding:utf-8
import sklearn 
import pandas as pd
import numpy as np


train = None
test = None

def read_data():
    global train, test
    train = pd.read_csv("../data/train.csv", header=0)
    test = pd.read_csv("../data/test.csv", header=0) 
    return
def feature_transform():
    merge_data = pd.concat([train, test]) 
    print(merge_data.info() )
    return

def train():
    return
def test():
    return
def predict():
    return

if __name__ == "__main__":
    read_data() 
    feature_transform() 
    train() 
    test() 
    predict() 

