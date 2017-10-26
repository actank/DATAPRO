#coding:utf-8

import sys
import os

os.system("cd ./prepare_data/train_data;rm -rf tmp;cat * > tmp")
os.system("cd ./prepare_data/val_data;rm -rf tmp;cat * > tmp")


f1 = open("./prepare_data/train_data/train.data", "w")
with open('./prepare_data/train_data/tmp') as f:
    for line in f:
        line = line.strip().split(",")[1:]
        out = ""
        for item in zip(line,[i for i in range(len(line))]):
            if item[1] == 0:
                out += item[0] + " "
            else:
                out += str(item[1]) + ":" + str(item[0]) + " "
        out = out.strip() + "\n"
        f1.write(out)
f1.close()

f1 = open("./prepare_data/val_data/val.data", "w")
with open('./prepare_data/val_data/tmp') as f:
    for line in f:
        line = line.strip().split(",")[1:]
        out = ""
        for item in zip(line,[i for i in range(len(line))]):
            if item[1] == 0:
                out += item[0] + " "
            else:
                out += str(item[1]) + ":" + str(item[0]) + " "
        out = out.strip() + "\n"
        f1.write(out)
f1.close()

os.system("cd ./prepare_data/test_data;cp ../val_data/val.data ./test.data")
os.system("cd ./prepare_data/test_data;cp ../val_data/tmp ./tmp")

