#coding:utf-8
import os
import sys
from datetime import *

feature_window_len = 10
label_window = "2014-12-17"

#暂时取滑动窗口为1天，方便调试，之后改为7进行提交
slide_window_len = 1

for i in range(slide_window_len):
    delta = timedelta(days=i)
    label_window_start = datetime.strptime("2014-12-17 00", '%Y-%m-%d %H') - delta
    label_window_end = datetime.strptime("2014-12-17 23", '%Y-%m-%d %H') - delta
    feature_window_start = label_window_start - timedelta(days=feature_window_len)
    label_window_start = label_window_start.strftime('%Y-%m-%d %H')
    label_window_end = label_window_end.strftime('%Y-%m-%d %H')
    feature_window_start = feature_window_start.strftime('%Y-%m-%d %H')
    os.system("hive -hiveconf train_table=%s -hiveconf l_w_s=%s -hiveconf l_w_e=%s -hiveconf f_w=%s -f get_feature.sql" % ('train_data_10', label_window_start, label_window_end, feature_window_start))

