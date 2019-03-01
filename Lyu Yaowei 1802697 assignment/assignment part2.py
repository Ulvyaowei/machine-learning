#!/usr/bin/python
# _*_ coding:utf-8 _*_

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import csv

df1 = pd.read_csv('CE802_Ass_2018_Data.csv')
df2 = pd.read_csv('CE802_Ass_2018_Test.csv')

data_test = []
data_test2 = []

data_raw = df1.values
data_target = df1.Class.values

data_raw_pre = MinMaxScaler().fit(data_raw)
data_raw_pre = data_raw_pre.transform(data_raw)
i = 0
while i < len(df1):
    data_test.append(list(data_raw_pre[i][0:-1]))
    i+=1

k = 0
data_raw2 = df2.values
data_raw2_pre = MinMaxScaler().fit_transform(data_raw2)
while k < len(df2):
    data_test2.append(list(data_raw2_pre[k][0:-1]))
    k+=1

svm = SVC(gamma=1.,C=1000.)


svm.fit(data_test,data_target)
svm_predict = svm.predict(data_test2)


dataframe = pd.DataFrame({'Class':svm_predict})
print(dataframe)
data_new = []
j = 0
while j < len(df1):
    data_new.append(list(data_raw2[j][0:-1]))
    data_new[j].append(svm_predict[j])
    j+=1
print(data_new)

with open('CE802_Ass_2018_Test.csv','w',newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["F1"],["F2"],["F3"],["F4"],["F5"],["F6"],["F7"],["F8"],["F9"],["F10"],["F11"],["F12"],["F13"],["F14"],["Class"])
    writer.writerows(data_new)




