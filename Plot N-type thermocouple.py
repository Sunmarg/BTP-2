# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 17:11:06 2021

@author: Sunmarg Das
"""

import pandas as pd
df = pd.read_excel (r'Thermocouple_data.xlsx',sheet_name='N_Type')
df=df[df['CI'].notna()]
df['Temp_Year']='200_2014'
df['Year']=2014
k=0
for i in range(0,len(df['Temp_Year'])):
    if (k<6):
        year='2014'
    elif (k<12):
        year='2016'
    elif (k<18):
        year='2019'
    else:
        year='2021'
    s=str(df['Temp'][i])+"_"+str(year)
    df['Temp_Year'][i]=s
    df['Year'][i]=int(year)
    k=k+1
    
d1=pd.DataFrame()
d1['XGBOOST']=X_pred
d1['Linear Regression']=pred_1
d1['Pol Regression']=pred_2
d1['Temp_Year']=df["Temp_Year"].iloc[-5:].values
d1.set_index('Temp_Year',inplace =True)
from matplotlib import pyplot as plt
%matplotlib inline

data=df.copy()
data.drop(['CI','Temp'],axis=1,inplace=True)
import seaborn as sns
sns.scatterplot(x="Temp_Year", y="Error", data=df)

plt.figure(figsize=(9,4))
plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.plot(df['Temp_Year'],df['Error'])
plt.tight_layout()
plt.xlabel(xlabel="Temp_year")
plt.ylabel(ylabel="Error")
plt.show()

data_1=df.drop(['CI','Temp_Year'],axis=1)
sns.boxplot(x='Year', y='Error', data=data_1)


plt.figure(figsize=(9,4))
plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.plot(d1['XGBOOST'])
plt.plot(d1['Linear Regression'])
plt.plot(d1['Pol Regression'])
plt.plot(y_test)
plt.tight_layout()
plt.legend(["XGBOOST","Linear Regression",'Pol Regression','Actual Interval'])
plt.ylabel(ylabel="Calibration Interval")
plt.xlabel(xlabel="Temp_Year")
plt.title("Comparision of Predicted Intervals with Actual Calibration Interval")
plt.show()