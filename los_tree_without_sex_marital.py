#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:23:54 2019

@author: zhangqiang
"""

import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image
import pydotplus
from sklearn.model_selection import GridSearchCV 
import matplotlib.pyplot as plt 
import datetime

def MABSE(true_value,prediction):
    true_value = np.array(true_value)
    prediction = np.array(prediction)
    #print(prediction)
    #print(true_value)
    #print(prediction-true_value)
    return np.mean(np.abs(prediction-true_value))

X_train = pd.read_csv('../data/X_train_los_tree_allFeatures.csv')
X_test = pd.read_csv('../data/X_test_los_tree_allFeatures.csv')
y_train = pd.read_csv('../data/y_train_los_tree_allFeatures.csv')
y_test = pd.read_csv('../data/y_test_los_tree_allFeatures.csv')

columns_name = list(X_train.columns)

del(columns_name[1:7])# delete columns containing sex and marital


X_train = X_train[columns_name]
X_test = X_test[columns_name]

"""
#select best paras
entropy_thresholds = np.linspace(0, 1, 100)
gini_thresholds = np.linspace(0, 0.2, 100)


param_grid = {
        'max_depth': np.arange(2,10),
        'min_samples_split': np.arange(2,30,2)
        }

clf = GridSearchCV(tree.DecisionTreeRegressor(), param_grid, cv=3, n_jobs = 3,scoring='neg_mean_absolute_error')
clf.fit(X_train, y_train)
print("best param:{0}\nbest score:{1}".format(clf.best_params_, clf.best_score_))
print(clf.best_params_)

y_pret = clf.predict(X_test)
print(y_pret)
print(y_test)
print(type(y_pret),type(y_test))

"""
tree_regression = tree.DecisionTreeRegressor(max_depth=3,min_samples_split=28)
time_start= datetime.datetime.now()
tree_regression.fit(X_train,y_train)
time_end = datetime.datetime.now()
print(time_end-time_start)
y_pret = tree_regression.predict(X_test)

error = MABSE(y_test.values.T[0],y_pret)
#print(y_test,y)
print(error)

"""
#visualization
dot_data = tree.export_graphviz(tree_regression, out_file=None,
                               feature_names=list(X_train.columns),
                               class_names=['inpatient','day case'],
                               filled=True, rounded=True,
                               special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

img = Image(graph.create_png())

graph.write_png("../fig/los_tree_allFeature.png")
"""
