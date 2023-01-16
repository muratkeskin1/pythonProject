# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 15:43:09 2022

@author: SerkanSavas
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Veri Kümesi Konumu
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Veri Kümesine ait Sütün Adları Ataması
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

iris_data = pd.read_csv(url, names=names)

iris_data.head()

#Veri Ön-işleme
X = iris_data.iloc[:, 0:4]
y = iris_data.select_dtypes(include=[object])
y.head()

y.Class.unique()


lab_enc = LabelEncoder()
y = y.apply(lab_enc.fit_transform)
y.Class.unique()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Model kurulumu
nn = MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10), max_iter=1000, verbose=1, activation='relu')
nn.fit(X_train, y_train)

#Tahminler
y_pred = nn.predict(X_test)

#Sonuçlar

print('Doğruluk oranı: %.2f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
