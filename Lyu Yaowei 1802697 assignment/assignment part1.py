#!/usr/bin/python
# _*_ coding:utf-8 _*_


import pandas as pd
#import the K-NN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree,svm
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import numpy as np



df = pd.read_csv('CE802_Ass_2018_Data.csv')

data_test = []

data_raw = df.values
data_target = df.Class.values
data_raw_pre = min_max_sclar = MinMaxScaler().fit_transform(data_raw)

i = 0
while i < len(df):
    data_test.append(list(data_raw_pre[i][0:-1]))
    i+=1


tree = tree.DecisionTreeClassifier(criterion='entropy')


clf = svm.SVC(gamma=0.01,C=10.)
Cs = np.logspace(-1,3,9)
Gs = np.logspace(-7,-0,8)
clf = GridSearchCV(estimator=clf,param_grid=dict(C = Cs,gamma = Gs),n_jobs=1)

score = []
clf.fit(data_test,data_target)
score.append(clf.score(data_test,data_target))
print(np.mean(score),clf.best_estimator_.C,clf.best_estimator_.gamma)



knn_best_score = 0
k = 1
while k<20:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data_test,data_target)
    knn_score = np.mean(cross_val_score(knn,data_test,data_target,cv = 5,scoring='accuracy'))
    if knn_score > knn_best_score:
        knn_best_score = knn_score
        knn_best_parameters = {'n_neighbors':k}
    k+=1
print(knn_best_score,knn_best_parameters)



tree_scores = cross_val_score(tree,data_test,data_target,cv=5,scoring = 'accuracy')
print(np.mean(tree_scores))




