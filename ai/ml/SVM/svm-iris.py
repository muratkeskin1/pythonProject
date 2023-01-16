# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 16:02:10 2022

@author: SerkanSavas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


url_name = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# Veri setinde bulunan öznitelik sütünlarının isimleri
columnnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# Pandas kütüphanesini kullanarak veriyi okuyoruz
irisdata = pd.read_csv(url_name, names=columnnames)
xdata = irisdata.drop('Class', axis=1)
ydata = irisdata['Class']


# Veriyi eğitim ve test veriseti olarak bölüyoruz
xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size = 0.2)


# Polynomial Kernel
svc_siniflandirici = SVC(kernel='poly', degree=8)
svc_siniflandirici.fit(xtrain, ytrain)
ypred = svc_siniflandirici.predict(xtest) # Sınıflandırıcı test ediliyor
print(confusion_matrix(ytest, ypred)) # Performans sonuçları yazdırılıyor
print(classification_report(ytest, ypred))


# Gaussian Kernel
svc_siniflandirici= SVC(kernel='rbf')
svc_siniflandirici.fit(xtrain, ytrain)
ypred = svc_siniflandirici.predict(xtest)
print(confusion_matrix(ytest, ypred))
print(classification_report(ytest, ypred))


#Sigmoid Kernel
svc_siniflandirici = SVC(kernel='sigmoid')
svc_siniflandirici.fit(xtrain, ytrain)
ypred = svc_siniflandirici.predict(xtest)
print(confusion_matrix(ytest, ypred))
print(classification_report(ytest, ypred))
