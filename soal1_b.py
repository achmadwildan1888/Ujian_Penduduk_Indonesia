import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
from sklearn.model_selection import train_test_split 

missing_values = ["n/a", "na", "-","NaN"]
data1 = pd.read_excel('indo_12_1.xls',na_values=missing_values,skiprows=3,skipfooter = 2)
data1.columns=['provinsi','1971','1980','1990','1995','2000','2010']
data1.dropna(inplace = True)

indonesia=data1['provinsi'][data1['2010']==data1['2010'].max()] 
dfindo=indonesia.index.tolist()[0] 
besar = data1.nlargest(2, "2010") 
dfjabar=besar['provinsi'].index.tolist()[1] 
kecil=data1['provinsi'][data1['1971']==data1['1971'].min()]
dfbengkulu=kecil.index.tolist()[0] 

x=np.array([1971,1980,1990,1995,2000,2010])
xtrain=x.reshape(-1,1)
kolom=['1971','1980','1990','1995','2000','2010']
y_indonesia=data1.loc[dfindo,kolom]
y_jabar=data1.loc[dfjabar,kolom]
y_bengkulu=data1.loc[dfbengkulu,kolom]

from sklearn import linear_model
model = linear_model.LinearRegression()

model.fit(xtrain,y_indonesia)
yindonesia=model.predict(xtrain)
print('Prediksi jumlah penduduk Indonesia di tahun 2050:', int(round(model.predict([[2050]])[0])))

model.fit(xtrain,y_jabar)
yjabar=model.predict(xtrain)
print('Prediksi jumlah penduduk Jawa Barat di tahun 2050:', int(round(model.predict([[2050]])[0])))


model.fit(xtrain,y_bengkulu)
ybengkulu=model.predict(xtrain)
print('Prediksi jumlah penduduk Bengkulu di tahun 2050:', int(round(model.predict([[2050]])[0])))

from matplotlib import style
plt.figure(figsize=(14,7))
style.use('seaborn')
plt.plot(xtrain,y_jabar,color='green',label=data1.provinsi[dfjabar])
plt.plot(xtrain,y_bengkulu,color='blue',label=data1.provinsi[dfbengkulu])
plt.plot(xtrain,y_indonesia,color='red',label=data1.provinsi[dfindo])
plt.plot(xtrain,yindonesia,'y-',label='Best Fit Line')
plt.plot(xtrain,yjabar,'y-')
plt.plot(xtrain,ybengkulu,'y-')
plt.legend()
plt.scatter(xtrain,yindonesia,color='red')
plt.scatter(xtrain,yjabar,color='green')
plt.scatter(xtrain,ybengkulu,color='blue')
plt.grid(True)
plt.ylabel('Jumlah penduduk (ratus juta jiwa)',fontsize=10)
plt.xlabel('Tahun',fontsize=10)
plt.title('Jumlah Penduduk INDONESIA (1971-2010)',fontsize=12)
plt.show()

