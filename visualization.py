#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:42:50 2019

@author: zhangqiang
"""
import pandas as pd
import matplotlib.pyplot as plt  
from matplotlib.backends.backend_pdf import PdfPages

data = pd.read_csv('../data/data.csv')

inpatient = data.loc[data.fake_ipdc=='I']
day_case = data.loc[data.fake_ipdc=='D']

#gender
num_inpatient = len(inpatient)
num_inpatient_sex = {}
num_day_case = len(day_case)
num_day_case_sex = {}
sex_index = {'1':'male','2':'female'}
sex_status = [1,2]
for i in sex_status:
    num_inpatient_sex[sex_index[str(i)]] = len(inpatient.loc[inpatient.fake_sex==i])
    num_day_case_sex[sex_index[str(i)]] = len(day_case.loc[day_case.fake_sex==i])

#age
num_inpatient_age = {}
num_day_case_age = {}
for i in range(min(data.fake_age),max(data.fake_age),10):
    num_inpatient_age[str(i)+'-'+str(i+9)] = len(inpatient.loc[inpatient.fake_age>=i].loc[inpatient.fake_age<i+10])
    num_day_case_age[str(i)+'-'+str(i+9)] = len(day_case.loc[day_case.fake_age>=i].loc[day_case.fake_age<i+10])

#marital
num_inpatient_marital = {}
num_day_case_marital = {}
marital_index = {'1':'Never married','2':'Married',
                 '3':'Widowed','8':'Other','9':'Not Known '}
marital_status = [1,2,3,8,9]
for i in marital_status:
    num_inpatient_marital[marital_index[str(i)]] = len(inpatient.loc[inpatient.fake_marital==i])
    num_day_case_marital[marital_index[str(i)]] = len(day_case.loc[day_case.fake_marital==i])
#print(num_inpatient_marital) 

#admission
num_inpatient_admission = {}
num_day_case_admission = {}
admission_index = {10:'Routine',20:'Urgent',30:'Emergency',40:'Other'}
admission_status = [10,20,30,40]
for i in admission_status:
    num_inpatient_admission[admission_index[i]] = len(inpatient.loc[inpatient.fake_admission>=i].loc[inpatient.fake_admission<i+10])
    num_day_case_admission[admission_index[i]] = len(day_case.loc[day_case.fake_admission>=i].loc[day_case.fake_admission<i+10])


plt.rcParams['figure.figsize'] = (16, 24)
fig = plt.figure()

plt.subplot(3,2,1)
plt.title('age - inpatient',fontsize = 15)
plt.pie(num_inpatient_age.values(),labels=num_inpatient_age.keys(),autopct='%1.2f%%') 
plt.tight_layout( pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.subplot(3,2,2)
plt.title('age - day case',fontsize = 15)
plt.pie(num_day_case_age.values(),labels=num_day_case_age.keys(),autopct='%1.2f%%')
plt.subplot(3,2,3)
plt.title('sex - inpatient',fontsize = 15)
plt.pie(num_inpatient_sex.values(),labels=num_inpatient_sex.keys(),autopct='%1.2f%%') 
plt.subplot(3,2,4)
plt.title('sex - day case',fontsize = 15)
plt.pie(num_day_case_sex.values(),labels=num_day_case_sex.keys(),autopct='%1.2f%%') 
plt.subplot(3,2,5)
plt.title('marital status - inpatient',fontsize = 15)
plt.pie(num_inpatient_marital.values(),labels=num_inpatient_marital.keys(),autopct='%1.2f%%') 
plt.subplot(3,2,6)
plt.title('marital status - day case',fontsize = 15)
plt.pie(num_day_case_marital.values(),labels=num_day_case_marital.keys(),autopct='%1.2f%%') 


pdf = PdfPages('../fig/fig_admission_demographics.pdf')
pdf.savefig()
plt.close()
pdf.close()
"""
#plt.show()  
"""
plt.rcParams['figure.figsize'] = (16, 8)
fig = plt.figure()
plt.subplot(1,2,1)
plt.title('admission - inpatient',fontsize = 15)
plt.pie(num_inpatient_admission.values(),labels=num_inpatient_admission.keys(),autopct='%1.2f%%') 
plt.tight_layout( pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.subplot(1,2,2)
plt.title('admission - day case',fontsize = 15)
plt.pie(num_day_case_admission.values(),labels=num_day_case_admission.keys(),autopct='%1.2f%%')
pdf = PdfPages('../fig/fig_admission_admission.pdf')
pdf.savefig()
plt.close()
pdf.close()