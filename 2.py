#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:31:33 2019

@author: zhangqiang
"""

import pandas as pd

data = pd.read_csv('../data/data.csv')

inpatients = data.loc[data.fake_ipdc == 'I']
day_case = data.loc[data.fake_ipdc == 'D']
print(len(inpatients.loc[inpatients.fake_sex==1]),len(inpatients.loc[inpatients.fake_sex==2]))
print(len(day_case.loc[day_case.fake_sex==1]),len(day_case.loc[day_case.fake_sex==2]))

print(len(inpatients.loc[inpatients.fake_sex==1]),len(inpatients.loc[inpatients.fake_sex==2]))
print(len(day_case.loc[day_case.fake_sex==1]),len(day_case.loc[day_case.fake_sex==2]))

print(len(data))