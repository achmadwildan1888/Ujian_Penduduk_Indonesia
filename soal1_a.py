#no.1
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

#kalkulasi
indonesia=data1['provinsi'][data1['2010']==data1['2010'].max()] 
dfindo=indonesia.index.tolist()[0] 
dfjabar=data1['provinsi'].index.tolist()[1] 
kecil=data1['provinsi'][data1['1971']==data1['1971'].min()] 
dfbengkulu=kecil.index.tolist()[0] 

index_x=[1971,1980,1990,1995,2000,2010]
kolom=['1971','1980','1990','1995','2000','2010']

y_indonesia=data1.loc[dfindo,kolom]
y_jabar=data1.loc[dfjabar,kolom]
y_bengkulu=data1.loc[dfbengkulu,kolom]


from matplotlib import style
plt.figure(figsize=(8,5))
style.use('seaborn')
plt.plot(index_x,y_indonesia,color='green',label=data1.provinsi[dfindo])
plt.plot(index_x,y_jabar,color='blue',label=data1.provinsi[dfjabar])
plt.plot(index_x,y_bengkulu,color='red',label=data1.provinsi[dfbengkulu])
plt.legend()
plt.scatter(index_x,y_indonesia,color='green')
plt.scatter(index_x,y_jabar,color='blue')
plt.scatter(index_x,y_bengkulu,color='red')
plt.grid(True)
plt.ylabel('Jumlah penduduk (ratus juta jiwa)',fontsize=10)
plt.xlabel('Tahun',fontsize=10)
plt.title('Jumlah Penduduk INDONESIA (1971-2010)',fontsize=12)
plt.show()