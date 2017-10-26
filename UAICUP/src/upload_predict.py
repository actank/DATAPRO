#coding:utf-8

import sys
import os

with open('./prepare_data/test_data/tmp') as f1, open('./pred.txt') as f2, open('./prepare_data/submit_data/predict.data', 'w') as f3:
    for line in f1:
        line = line.strip()
        line2 = f2.readline().strip()
        if line == "":
            break
        elif line2 == "":
            break
        id = line.split(",")[0]
        f3.write("%s,%s\n" % (id, line2))
        
