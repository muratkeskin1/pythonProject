# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 18:22:24 2022

@author: SerkanSavas
"""

# Kütüphanelerin Yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Veri Setinin Yüklenmesi
dataset = pd.read_csv('Position_Salaries.csv')
dataset.head()
 
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
 
# Model Kurulumu 10 Ağaç Sayısı
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X,y)

#Grafiksel Gösterimi
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1) 
plt.scatter(X,y, color='red') #gerçek noktaların çizilmesi
plt.plot(X_grid, regressor.predict(X_grid),color='blue') #tahmin noktalarını çizdirme
plt.title("Truth or Bluff(Random Forest - Smooth - 10)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Model Tahmin Sonucu 
y_pred=regressor.predict([[6.5]])
y_pred
 
# Model Kurulumu 100 Ağaç Sayısı
regressor2 = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor2.fit(X,y)
 
#Grafiksel Gösterimi 
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1) 
plt.scatter(X,y, color='red')  #gerçek noktaların çizilmesi 
plt.plot(X_grid, regressor2.predict(X_grid),color='blue') #tahmin noktalarını çizdirme
plt.title("Truth or Bluff(Random Forest - Smooth - 100)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
 
#Model Tahmin Sonucu 
y_pred2=regressor2.predict([[6.5]])
y_pred2

# Model Kurulumu 300 Ağaç Sayısı
regressor3 = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor3.fit(X,y)

#Grafiksel Gösterimi 
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)   
plt.scatter(X,y, color='red') #gerçek noktaların çizilmesi
plt.plot(X_grid, regressor3.predict(X_grid),color='blue') #tahmin noktalarını çizdirme
plt.title("Truth or Bluff(Random Forest - Smooth - 300)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

y_pred3=regressor3.predict([[6.5]])
y_pred3