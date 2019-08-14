#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:29:48 2019

@author: zhangqiang
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import datetime

X_train = pd.read_csv('../data/X_train_logistic_allFeatures.csv')
X_test = pd.read_csv('../data/X_test_logistic_allFeatures.csv')
y_train = pd.read_csv('../data/y_train_logistic_allFeatures.csv')
y_test = pd.read_csv('../data/y_test_logistic_allFeatures.csv')

"""
columns_name = list(X_train.columns)
del(columns_name[1:7])# delete columns containing sex and marital

X_train = X_train[columns_name]
X_test = X_test[columns_name]
"""
lr = LogisticRegression(solver = "liblinear",max_iter=1000,
                        C = 0.3138888888888889,penalty='l1')

"""
param_test = {
        'penalty': ['l1','l2'],
        'C': np.linspace(0.05, 1, 19)
    }

gsearch = GridSearchCV(lr, param_grid=param_test, scoring='neg_mean_squared_error',
                       n_jobs=3, cv=3)


gsearch.fit(X_train,y_train.values.T[0])

print('*'*10+'Finished'+'*'*10)
print('*'*50)
print(gsearch.best_params_)
y_pret = gsearch.predict(X_test)
"""
time_start = datetime.datetime.now()
lr.fit(X_train,y_train.values.T[0])
time_end = datetime.datetime.now()
print(time_end-time_start)
y_pret = lr.predict(X_test)

for i in np.arange(0,len(y_pret),1):
    if (y_pret[i]>0.5):
        y_pret[i] = 1
    else:
        y_pret[i] = 0

print(sum(y_test.values.T[0]==y_pret)/len(y_pret))

print('*'*50)