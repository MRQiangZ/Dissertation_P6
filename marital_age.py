#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:06:19 2019

@author: zhangqiang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from matplotlib.backends.backend_pdf import PdfPages

data = pd.read_csv('../data/data.csv')

marital = []
age = []
marital_index = {1:'Never married',2:'Married',
                 3:'Widowed',8:'Other',9:'Not Known '}
marital_status = [1,2,3,8,9]
print(len(data))
for (index,cont) in data.iterrows():
    #print(i)
    marital.append(marital_index[cont.fake_marital])
    age.append(cont.fake_age)

plt.rcParams['figure.figsize'] = (9, 5)
plt.xlabel('Marital status')
plt.ylabel('Age')
plt.scatter(marital,age)    

pdf = PdfPages('../fig/fig_marital_age.pdf')
pdf.savefig()
plt.close()
pdf.close()