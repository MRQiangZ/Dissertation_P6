#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:00:34 2019

@author: zhangqiang
"""

import numpy as np
import pandas as pd
import datetime
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV

def MABSE(true_value,prediction):
    true_value = np.array(true_value)
    prediction = np.array(prediction)
    return np.mean(np.abs(prediction-true_value))

X_train = pd.read_csv('../data/X_train_los_xgboost_allFeatures.csv')
X_test = pd.read_csv('../data/X_test_los_xgboost_allFeatures.csv')
y_train = pd.read_csv('../data/y_train_los_xgboost_allFeatures.csv')
y_test = pd.read_csv('../data/y_test_los_xgboost_allFeatures.csv')

clf = XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,  # [默认是0.3]学习率类似，调小能减轻过拟合，经典值是0.01-0.2
        gamma=0,  # 在节点分裂时，只有在分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。这个参数值越大，算法越保守。
        subsample=0.8,  # 随机采样比例，0.5-1 小欠拟合，大过拟合
        colsample_bytree=0.8,  # 训练每棵树时用来训练的特征的比例
        reg_alpha=1,  # [默认是1] 权重的L1正则化项
        reg_lambda=1,  # [默认是1] 权重的L2正则化项
        max_depth=10,  # [默认是6] 树的最大深度，这个值也是用来避免过拟合的3-10
        min_child_weight=1,  # [默认是1]决定最小叶子节点样本权重和。当它的值较大时，可以避免模型学习到局部的特殊样本。但如果这个值过高，会导致欠拟合。
        n_jobs=1
)
"""
dtrain = xgb.DMatrix(X_train, y_train)
xgb_params = clf.get_xgb_params()
cvresult = xgb.cv(xgb_params, dtrain, nfold=5, num_boost_round=2000,
                      early_stopping_rounds=50)
#clf_xgb = xgb.train(xgb_params, dtrain, num_boost_round=cvresult.shape[0])
#fscore = clf_xgb.get_fscore()
#print(cvresult.shape[0], fscore)
print(cvresult.shape[0])
"""
clf.set_params(n_estimators=10)
"""
param_test1 = {
        'max_depth': [i for i in range(3, 12, 2)],
        'min_child_weight': [i for i in range(1, 10, 2)]
    }
best_max_depth = 3
best_min_child_weight = 9
param_test2 = {
        'max_depth': [i for i in range(3, 12, 2)],
        'min_child_weight': [i for i in range(1, 10, 2)]
    }
"""
clf.set_params(max_depth=3,min_child_weight=9)
"""
param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 10, 2)]
    }
best_gamma = 0
param_test4 = {
        'gamma': [best_gamma,best_gamma+0.1]
    }
"""
clf.set_params(gamma = 0)
"""
param_test5 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
"""
"""
best_subsample = 0.9
best_colsample_bytree = 0.9
param_test6 = {
        'subsample': [best_subsample-0.05,best_subsample,best_subsample+0.05],
        'colsample_bytree': [best_colsample_bytree-0.05,best_colsample_bytree,best_colsample_bytree+0.05]
}
"""
#clf.set_params(b_subsample = 0.95,b_colsample_bytree = 0.95)
"""
param_test7 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 2],
        'reg_lambda': [1e-5, 1e-2, 0.1, 1, 2]
    }
"""
#clf.set_params(reg_alpha = 1e-5,reg_lambda = 1e-5)

"""
gsearch = GridSearchCV(clf, param_grid=param_test6, scoring='neg_mean_squared_error',n_jobs=3, cv=3)


time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
gsearch.fit(X_train,y_train)
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)
print(gsearch.best_params_)
print(gsearch.best_score_)
y_pret = gsearch.predict(X_test)
#print(RMSE(y_pret,y_test.values.T[0]))
"""
time_start= datetime.datetime.now()
clf.fit(X_train,y_train)
time_end = datetime.datetime.now()
print(time_end-time_start)
y_pret = clf.predict(X_test)
print(MABSE(y_test.values.T[0],y_pret))


