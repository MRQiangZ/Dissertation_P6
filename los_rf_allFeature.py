#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:10:12 2019

@author: zhangqiang
"""

import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def MABSE(true_value,prediction):
    true_value = np.array(true_value)
    prediction = np.array(prediction)
    return np.mean(np.abs(prediction-true_value))

X_train = pd.read_csv('../data/X_train_los_xgboost_allFeatures.csv')
X_test = pd.read_csv('../data/X_test_los_xgboost_allFeatures.csv')
y_train = pd.read_csv('../data/y_train_los_xgboost_allFeatures.csv')
y_test = pd.read_csv('../data/y_test_los_xgboost_allFeatures.csv')

clf = RandomForestRegressor(
            n_estimators=2,             # 学习器个数
            criterion='mse',             # 评价函数
            max_depth=None,              # 最大的树深度，防止过拟合
            min_samples_split=2,         # 根据属性划分节点时，每个划分最少的样本数
            min_samples_leaf=1,          # 最小叶子节点的样本数，防止过拟合
            max_features='auto',         # auto是sqrt(features)还有 log2 和 None可选
            max_leaf_nodes=None,         # 叶子树的最大样本数
            bootstrap=True,              # 有放回的采样
            min_weight_fraction_leaf=0,
            n_jobs=3)                   # 同时用多少个进程训练
"""
param_test1 = {
        'n_estimators': [i for i in range(100, 201, 20)]
    }
"""
clf.set_params(n_estimators=160)
"""
param_test2_1 = {
        'max_depth': [45,50,55,60],
        'min_samples_split' : np.arange(2,5,2),
        'min_samples_leaf' : np.arange(1,5,2)
}
"""
clf.set_params(min_samples_split=2,min_samples_leaf= 3)
"""
max_d = 60
param_test2_2 = {
        'max_depth': [max_d-2, max_d, max_d+2]
    }
"""
clf.set_params(max_depth = 60)
"""
param_test3_1 = {
        'max_features': [0.1,0.3,0.5, 0.7,0.9]
    }


max_f = 0.1
param_test3_2 = {
        'max_features': [ max_f, max_f+0.1]
    }
"""

clf.set_params(max_features = 0.1)

"""
gsearch = GridSearchCV(clf, param_grid=param_test3_1, scoring='neg_mean_squared_error',n_jobs=1, cv=3)
time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
gsearch.fit(X_train,y_train.values.T[0])
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)

print(gsearch.best_params_)
print(gsearch.best_score_)

y_pret = gsearch.predict(X_test)
"""
time_start= datetime.datetime.now()
clf.fit(X_train,y_train.values.T[0])
time_end = datetime.datetime.now()
print(time_end-time_start)
y_pret = clf.predict(X_test)
print(MABSE(y_test.values.T[0],y_pret))