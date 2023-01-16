# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 11:51:40 2022

@author: SerkanSavas
"""

#Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Veri Kümesi Hazırlama
iris=pd.read_csv('Iris.csv') #veri kümesi yolu
iris.head()
iris.tail()
iris.describe(include='all')
iris.drop(columns="Id", inplace=True)

#Veri Kümesi İnceleme
sns.relplot(x='SepalLengthCm', y='SepalWidthCm', data=iris, hue='Species', style='Species')
plt.show()

sns.relplot(x='PetalLengthCm', y='PetalWidthCm', data=iris, hue='Species', style='Species')
plt.show()

#Veri Ön İşleme
X=iris.iloc[:,0:4].values 
y=iris.iloc[:,4].values
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)


#algoritma Uygulaması
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
acc_gaussian = gaussian.score(X_train, y_train)
print("Eğitim Doğruluğu: %.2f " %acc_gaussian)
Y_pred = gaussian.predict(X_test) 
acc_nb=accuracy_score(y_test,Y_pred)
print("Test Doğruluğu: %.2f " %acc_nb)

#Karışıklık Matrisi
cm = confusion_matrix(y_test, Y_pred)
precision = precision_score(y_test, Y_pred, average='weighted')
recall =  recall_score(y_test, Y_pred, average='weighted')
f1 = f1_score(y_test, Y_pred, average='weighted')
print('Confusion matrix\n ', cm)
print('Precision: %.2f ' %precision)
print('Recall: %.2f ' %recall)
print('f1-score: %.2f ' %f1)
