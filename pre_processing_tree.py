#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:14:38 2019

@author: zhangqiang
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/data.csv')

data_one_hot = data[['fake_age','fake_sex']]


#marital status
marital_index = {1:'Never married',2:'Married',
                 3:'Widowed',8:'Other',9:'Not Known'}
marital_status = [1,2,3,8,9]
for i in marital_status:
    data['fake_marital'].loc[data.fake_marital==i] = marital_index[i]
data_one_hot_marital = pd.get_dummies(data['fake_marital'],
                                      prefix='fake_marital',prefix_sep='_')
data_one_hot = pd.merge(data_one_hot,data_one_hot_marital,left_index=True,right_index=True)

#admission
admission_index = {10:'Routine',20:'Urgent',
                 30:'Emergency',40:'Other'}
admission_status = [10,20,30,40]
admission_copy = pd.DataFrame(data.fake_admission.values,columns=['fake_admission'])
for i in admission_status:
    data['fake_admission'].loc[admission_copy.fake_admission>=i] = admission_index[i]
data_one_hot_admission = pd.get_dummies(data['fake_admission'],
                                        prefix='fake_admission',prefix_sep='_')
data_one_hot = pd.merge(data_one_hot,data_one_hot_admission,left_index=True,right_index=True)

#specialties
data_one_hot_spec = pd.get_dummies(data['fake_spec'],
                                   prefix='fake_spec',prefix_sep='_')
data_one_hot = pd.merge(data_one_hot,data_one_hot_spec,left_index=True,right_index=True)

#laebl(ipdc)
data['fake_ipdc'].loc[data.fake_ipdc=='I'] = 'inpatient'
data['fake_ipdc'].loc[data.fake_ipdc=='D'] = 'day case'
label = pd.DataFrame(data.fake_ipdc.values,columns=['fake_ipdc'])



#split into training and test set
X_train, X_test, y_train, y_test = train_test_split( data_one_hot, label, test_size=0.3)

X_train.to_csv('../data/X_train_tree_allFeatures.csv',index=False)
X_test.to_csv('../data/X_test_tree_allFeatures.csv',index=False)
y_train.to_csv('../data/y_train_tree_allFeatures.csv',index=False)
y_test.to_csv('../data/y_test_tree_allFeatures.csv',index=False)






