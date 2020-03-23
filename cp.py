import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
dataset = pd.read_excel(open('T.xlsx','rb'))
dataset2=pd.read_excel(open('Test.xlsx','rb'))
dataset = dataset.fillna(method='ffill')
dataset2 = dataset2.fillna(method='ffill')
X = dataset[['Gender','Mode_transport','cases/1M','Deaths/1M','comorbidity','Age','Coma score','Pulmonary score','cardiological pressure','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose','FT/month']].values
y = dataset['Infect_Prob'].values
x_test=dataset2[['Gender','Mode_transport','cases/1M','Deaths/1M','comorbidity','Age','Coma score','Pulmonary score','cardiological pressure','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose','FT/month']].values
reg=LinearRegression(normalize=True)
reg.fit(X,y)
y_pred = reg.predict(x_test)
b={'ID':dataset2['people_ID'].values,'Infect_Prob':y_pred}
df =  pd.DataFrame(b)
df.to_csv('output.csv', sep=',', header=None, index=None)
