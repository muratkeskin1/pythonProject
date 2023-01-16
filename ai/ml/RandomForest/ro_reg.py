# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 18:04:00 2022

@author: SerkanSavas
"""

# Kütüphanelerin Yüklenmesi
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn import set_config

# Veri Setinin Yüklenmesi
X, y = make_regression(n_samples=5000, n_features=10)
print(X[0:2])
print(y[0:2])

#Veri Setinin Eğitim ve Test Veri Seti Olarak Ayrılması
X = scale(X)
y = scale(y)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.15)

#Model Kurulumu
rfr = RandomForestRegressor()
rfr.fit(Xtrain, ytrain)

score = rfr.score(Xtrain, ytrain)
print("R-squared:", score)

#Tahmin ve Doğruluk kontrolü
ypred = rfr.predict(Xtest)

mse = mean_squared_error(ytest, ypred)
print("MSE: ", mse)
print("RMSE: ", mse*(1/2.0))

#Grafiksel Gösterimi
x_ax = range(len(ytest))
plt.plot(x_ax, ytest, linewidth=1, label="original")
plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()
