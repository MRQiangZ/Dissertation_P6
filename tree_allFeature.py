#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:33:02 2019

@author: zhangqiang
"""
import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image
import pydotplus
from sklearn.model_selection import GridSearchCV 
import datetime

X_train = pd.read_csv('../data/X_train_tree_allFeatures.csv')
X_test = pd.read_csv('../data/X_test_tree_allFeatures.csv')
y_train = pd.read_csv('../data/y_train_tree_allFeatures.csv')
y_test = pd.read_csv('../data/y_test_tree_allFeatures.csv')

#X_train = X_train[['fake_age','fake_admission_Emergency','fake_admission_Routine','fake_admission_Urgent']]
#X_test = X_test[['fake_age','fake_admission_Emergency','fake_admission_Routine','fake_admission_Urgent']]

#select best paras
entropy_thresholds = np.linspace(0, 1, 100)
gini_thresholds = np.linspace(0, 0.2, 100)
"""
param_grid = [{'criterion': ['entropy'], 'min_impurity_decrease': entropy_thresholds},
              {'criterion': ['gini'], 'min_impurity_decrease': gini_thresholds},
              {'max_depth': np.arange(2,10)},
              {'min_samples_split': np.arange(2,30,2)}]
"""
"""
param_grid = {
        'criterion': ['gini'], 
        'min_impurity_decrease': gini_thresholds,
        'max_depth': np.arange(2,10),
        'min_samples_split': np.arange(2,30,2)
        }
"""
"""
clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=3, n_jobs = 3)
clf.fit(X_train, y_train)
print("best param:{0}\nbest score:{1}".format(clf.best_params_, clf.best_score_))
print(clf.best_params_)

y_pret = clf.predict(X_test)
print(sum(y_test.values.T[0]==y_pret)/len(y_pret))
"""

#tree
tree_admission = tree.DecisionTreeClassifier(criterion='gini',max_depth=6,
                                             min_impurity_decrease=0.00202020202020202,min_samples_split=2)
time_start = datetime.datetime.now()
tree_admission.fit(X_train,y_train)
time_end = datetime.datetime.now()
print(time_end-time_start)
y_pret = tree_admission.predict(X_test)
print(sum(y_test.values.T[0]==y_pret)/len(y_pret))

"""
#visualization
dot_data = tree.export_graphviz(tree_admission, out_file=None,
                               feature_names=list(X_train.columns),
                               class_names=['inpatient','day case'],
                               filled=True, rounded=True,
                               special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

img = Image(graph.create_png())

graph.write_pdf("../fig/tree_allFeature.pdf")
#graph.write_png("../fig/tree_allFeature.png")
"""