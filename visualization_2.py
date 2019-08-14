#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:12:11 2019

@author: zhangqiang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from matplotlib.backends.backend_pdf import PdfPages

data = pd.read_csv('../data/data.csv')

age_data = {}
for i in range(min(data.fake_age),max(data.fake_age),10):
    age_data[str(i)+'-'+str(i+9)] = data.loc[data.fake_age>=i].loc[data.fake_age<i+10]
#print(age_data)
inpatient_ratio_age = {}
los_mean_age = {}
los_std_age = {}
#percentile of los
los_25_age = {}
los_50_age = {}
los_75_age = {}
for (key,content) in age_data.items():
    inpatient_part = content.loc[content.fake_ipdc=='I']
    #obtain the ratio of inpatient in people with different ages
    inpatient_ratio_age[key] = len(inpatient_part)/len(content)
    #obtain the mean and std of inpatient with different ages
    inpatient_los = list(inpatient_part.fake_los)
    los_mean_age[key] = np.mean(inpatient_los)
    los_std_age[key] = np.std(inpatient_los)
    los_25_age[key] = np.percentile(inpatient_los,25)
    los_50_age[key] = np.percentile(inpatient_los,50)
    los_75_age[key] = np.percentile(inpatient_los,75)
print(inpatient_ratio_age)
print(los_mean_age)
print(los_std_age)
print(los_25_age)
print(los_50_age)
print(los_75_age)

sex_data = {}
sex_index = {1:'male',2:'female'}
sex_status = [1,2]
for i in sex_status:
    sex_data[sex_index[i]] = data.loc[data.fake_sex==i]

inpatient_ratio_sex = {}
los_mean_sex = {}
los_std_sex = {}
#percentile of los
los_25_sex = {}
los_50_sex = {}
los_75_sex = {}

for (key,content) in sex_data.items():
    inpatient_part = content.loc[content.fake_ipdc=='I']
    #obtain the ratio of inpatient in people with different ages
    inpatient_ratio_sex[key] = len(inpatient_part)/len(content)
    #obtain the mean and std of inpatient with different ages
    inpatient_los = list(inpatient_part.fake_los)
    los_mean_sex[key] = np.mean(inpatient_los)
    los_std_sex[key] = np.std(inpatient_los)
    los_25_sex[key] = np.percentile(inpatient_los,25)
    los_50_sex[key] = np.percentile(inpatient_los,50)
    los_75_sex[key] = np.percentile(inpatient_los,75)
    
print(inpatient_ratio_sex)
print(los_mean_sex)
print(los_std_sex)
print(los_25_sex)
print(los_50_sex)
print(los_75_sex)


marital_data = {}
marital_index = {1:'Never married',2:'Married',
                 3:'Widowed',8:'Other',9:'Not Known '}
marital_status = [1,2,3,8,9]
for i in marital_status:
    marital_data[marital_index[i]] = data.loc[data.fake_marital==i]
    
inpatient_ratio_marital = {}
los_mean_marital = {}
los_std_marital = {}
#percentile of los
los_25_marital = {}
los_50_marital = {}
los_75_marital = {}

for (key,content) in marital_data.items():
    inpatient_part = content.loc[content.fake_ipdc=='I']
    #obtain the ratio of inpatient in people with different ages
    inpatient_ratio_marital[key] = len(inpatient_part)/len(content)
    #obtain the mean and std of inpatient with different ages
    inpatient_los = list(inpatient_part.fake_los)
    los_mean_marital[key] = np.mean(inpatient_los)
    los_std_marital[key] = np.std(inpatient_los)
    los_25_marital[key] = np.percentile(inpatient_los,25)
    los_50_marital[key] = np.percentile(inpatient_los,50)
    los_75_marital[key] = np.percentile(inpatient_los,75)
    
print(inpatient_ratio_marital)
print(los_mean_marital)
print(los_std_marital)
print(los_25_marital)
print(los_50_marital)
print(los_75_marital)

admission_data = {}
admission_index = {10:'Routine',20:'Urgent',30:'Emergency',40:'Other'}
admission_status = [10,20,30,40]
for i in admission_status:
    admission_data[admission_index[i]] = data.loc[data.fake_admission>=i].loc[data.fake_admission<i+10]

inpatient_ratio_admission = {}
los_mean_admission = {}
los_std_admission = {}
los_25_admission = {}
los_50_admission = {}
los_75_admission = {}

for (key,cont) in admission_data.items():
    inpatient_part = cont.loc[cont.fake_ipdc=='I']
    if len(cont)==0:
        inpatient_ratio_admission[key] = 0
        los_mean_admission[key] = 0
        los_std_admission[key] = 0
        los_25_admission[key] = 0
        los_50_admission[key] = 0
        los_75_admission[key] = 0       
    else:
        inpatient_ratio_admission[key] = len(inpatient_part)/len(cont)
        inpatient_los = list(inpatient_part.fake_los)
        los_mean_admission[key] = np.mean(inpatient_los)
        los_std_admission[key] = np.std(inpatient_los)
        los_25_admission[key] = np.percentile(inpatient_los,25)
        los_50_admission[key] = np.percentile(inpatient_los,50)
        los_75_admission[key] = np.percentile(inpatient_los,75)
    
#start plot
#print(list(inpatient_ratio_age.keys),list(inpatient_ratio_age.values))
plt.rcParams['figure.figsize'] = (15, 5)


plt.subplot(1,3,1)
plt.ylim(0.5,1)
plt.plot(inpatient_ratio_age.keys(),inpatient_ratio_age.values())
plt.xticks(rotation=18) 
plt.title('Age')
plt.ylabel('Ratio')
plt.subplot(1,3,2)
plt.ylim(0.5,1)
plt.plot(inpatient_ratio_sex.keys(),inpatient_ratio_sex.values())
plt.xticks(rotation=18)
plt.title('Sex')
plt.ylabel('Ratio')
plt.subplot(1,3,3)
plt.ylim(0.5,1)
plt.plot(inpatient_ratio_marital.keys(),inpatient_ratio_marital.values())
plt.xticks(rotation=18)
plt.title('Marital Status')
plt.ylabel('Ratio')
#plt.show()

pdf_1 = PdfPages('../fig/fig_inpatient_ratio.pdf')
pdf_1.savefig()
plt.close()
pdf_1.close()

plt.rcParams['figure.figsize'] = (5, 5)
plt.subplot(1,1,1)
#plt.ylim(0.5,1)
plt.plot(inpatient_ratio_admission.keys(),inpatient_ratio_admission.values())
plt.xticks(rotation=18) 
plt.title('Admission')
plt.ylabel('Ratio')

pdf_1 = PdfPages('../fig/fig_inpatient_ratio_admission.pdf')
pdf_1.savefig()
plt.close()
pdf_1.close()

plt.rcParams['figure.figsize'] = (24, 8)
#print(np.array(list(los_mean_age.values()))-np.array(list(los_std_age.values())))
plt.subplot(1,3,1)
plt.ylim(-10,30)
plt.plot(los_mean_age.keys(),los_mean_age.values())
plt.plot(los_mean_age.keys(),np.array(list(los_mean_age.values()))-np.array(list(los_std_age.values())))
plt.plot(los_mean_age.keys(),np.array(list(los_mean_age.values()))+np.array(list(los_std_age.values())))
plt.xticks(rotation=18) 
plt.title('Age')
plt.ylabel('Length of stay')
plt.subplot(1,3,2)
plt.ylim(-10,30)
plt.plot(los_mean_sex.keys(),los_mean_sex.values())
plt.plot(los_mean_sex.keys(),np.array(list(los_mean_sex.values()))-np.array(list(los_std_sex.values())))
plt.plot(los_mean_sex.keys(),np.array(list(los_mean_sex.values()))+np.array(list(los_std_sex.values())))
plt.xticks(rotation=18) 
plt.title('Sex')
plt.ylabel('Length of stay')
plt.subplot(1,3,3)
plt.ylim(-10,30)
plt.plot(los_mean_marital.keys(),los_mean_marital.values())
plt.plot(los_mean_marital.keys(),np.array(list(los_mean_marital.values()))-np.array(list(los_std_marital.values())))
plt.plot(los_mean_marital.keys(),np.array(list(los_mean_marital.values()))+np.array(list(los_std_marital.values())))
plt.xticks(rotation=18) 
plt.title('Marital Status')
plt.ylabel('Length of stay')
#plt.show()

pdf_2 = PdfPages('../fig/fig_los_mean_std.pdf')
pdf_2.savefig()
plt.close()
pdf_2.close()

plt.rcParams['figure.figsize'] = (5, 5)
plt.subplot(1,1,1)
#plt.ylim(-10,30)
plt.plot(los_mean_admission.keys(),los_mean_admission.values())
plt.plot(los_mean_admission.keys(),np.array(list(los_mean_admission.values()))-np.array(list(los_std_admission.values())))
plt.plot(los_mean_admission.keys(),np.array(list(los_mean_admission.values()))+np.array(list(los_std_admission.values())))
plt.xticks(rotation=18) 
plt.title('Admission')
plt.ylabel('Length of stay')

pdf_2 = PdfPages('../fig/fig_los_mean_std_admission.pdf')
pdf_2.savefig()
plt.close()
pdf_2.close()

plt.rcParams['figure.figsize'] = (24, 8)
plt.subplot(1,3,1)
plt.ylim(0,12)
plt.plot(los_50_age.keys(),los_50_age.values())
plt.plot(los_25_age.keys(),los_25_age.values())
plt.plot(los_75_age.keys(),los_75_age.values())
plt.xticks(rotation=18) 
plt.title('Age')
plt.ylabel('Length of stay')
plt.subplot(1,3,2)
plt.ylim(0,12)
plt.plot(los_50_sex.keys(),los_50_sex.values())
plt.plot(los_25_sex.keys(),los_25_sex.values())
plt.plot(los_75_sex.keys(),los_75_sex.values())
plt.xticks(rotation=18) 
plt.title('Sex')
plt.ylabel('Length of stay')
plt.subplot(1,3,3)
plt.ylim(0,12)
plt.plot(los_50_marital.keys(),los_50_marital.values())
plt.plot(los_25_marital.keys(),los_25_marital.values())
plt.plot(los_75_marital.keys(),los_75_marital.values())
plt.xticks(rotation=18) 
plt.title('Marital Status')
plt.ylabel('Length of stay')
#plt.show()

pdf_3 = PdfPages('../fig/fig_los_percentile.pdf')
pdf_3.savefig()
plt.close()
pdf_3.close()

plt.rcParams['figure.figsize'] = (5, 5)
plt.subplot(1,1,1)
#plt.ylim(0,12)
plt.plot(los_50_admission.keys(),los_50_admission.values())
plt.plot(los_25_admission.keys(),los_25_admission.values())
plt.plot(los_75_admission.keys(),los_75_admission.values())
plt.xticks(rotation=18) 
plt.title('Admission')
plt.ylabel('Length of stay')

pdf_3 = PdfPages('../fig/fig_los_percentile_admission.pdf')
pdf_3.savefig()
plt.close()
pdf_3.close()




