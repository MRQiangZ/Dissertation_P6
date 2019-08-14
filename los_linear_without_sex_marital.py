#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:15:52 2019

@author: zhangqiang
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def MABSE(true_value,prediction):
    true_value = np.array(true_value)
    prediction = np.array(prediction)
    return np.mean(np.abs(prediction-true_value))

X_train = pd.read_csv('../data/X_train_los_xgboost_allFeatures.csv')
X_test = pd.read_csv('../data/X_test_los_xgboost_allFeatures.csv')
y_train = pd.read_csv('../data/y_train_los_xgboost_allFeatures.csv')
y_test = pd.read_csv('../data/y_test_los_xgboost_allFeatures.csv')

columns_name = list(X_train.columns)

del(columns_name[1:7])# delete columns containing sex and marital


X_train = X_train[columns_name]
X_test = X_test[columns_name]

linear = LinearRegression()
linear.fit(X_train,y_train)
y_pret = linear.predict(X_test)
print(MABSE(y_test.values.T[0],y_pret))
