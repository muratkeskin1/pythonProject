# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 20:06:10 2022

@author: SerkanSavas
"""

import pandas as pd
covid=pd.read_csv('qt_dataset.csv')
covid.head()

covid.info()
covid.describe()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('ggplot')

#Veri kümesinde alanlar arasındaki korelasyonu görmek için
korelasyon = np.round(covid.corr(),2)
fig, ax = plt.subplots(figsize = (14, 10))
sns.heatmap(ax = ax, \
            data = korelasyon, \
            annot = True, \
            cmap = "coolwarm", \
            vmin = -1, vmax= 1, center = 0)
ax.set_title("Korelasyon Matrisi Grafiği")
plt.show()

#Veri kümesinde sağlıklı ve hasta olan kayıtların yüzdelik olarak listelenmesi
round(covid.Result.value_counts()*100/len(covid), 2)


#Veri ön-işleme
X=covid[['Oxygen','PulseRate', 'Temperature']].values
y = covid['Result'].values	
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#K-En Yakın Komşu Sınıflandırıcı Kütüphanesi ekleme
from sklearn.neighbors import KNeighborsClassifier

# Eğitim ve test doğruluğunu saklamak için dizileri ayarlama
neighbors = np.arange(1,9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    # K komşuları ile bir knn sınıflandırıcısı kurulması
    knn = KNeighborsClassifier(n_neighbors=k)
    
# Modele uydurma
    knn.fit(X_train, y_train)
    
    # Eğitim setinde hesaplama doğruluğu
    train_accuracy[i] = knn.score(X_train, y_train)
    
    # Test setinde hesaplama doğruluğu
    test_accuracy[i] = knn.score(X_test, y_test)


# Grafiğin oluşturulması
plt.title('k-En Değişen Komşu Sayısı')
plt.plot(neighbors, test_accuracy, label='Test Doğruluğu')
plt.plot(neighbors, train_accuracy, label='Eğitim Doğruluğu')
plt.legend()
plt.xlabel('Komşu Sayısı')
plt.ylabel('Doğruluk')
plt.show()

# K komşuları ile bir knn sınıflandırıcısı kurulması
knn = KNeighborsClassifier(n_neighbors=5)
#modelin uygulanması
knn.fit(X_train,y_train)

knn.score(X_test,y_test)
y_pred=knn.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
