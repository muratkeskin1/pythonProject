# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:41:17 2022

@author: SerkanSavas
"""

#Kullanılan kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree
import graphviz 


#Veri seti indiriliyor
iris = load_iris()
X, y = iris.data, iris.target


#Veri setindeki başlıklar ve örnek veriler
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()


#Nitelikler, nitelik veri türleri, kayıt sayısı
df.info()


#İstatistiksel bilgiler
df.describe()


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)


#Ağacın değerlerle beraber görsel gösterimi
dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=iris.feature_names,  
                      class_names=iris.target_names,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# Verinin test ve eğitim verisi olarak bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Eğitim verisi kullanılarak ağaç oluşturulması
clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(X_train, y_train)

# Test verisi kullanılarak doğruluğunun ölçülmesi
from sklearn.metrics import accuracy_score
y_pred=clf2.predict(X_test)
accuracy_score(y_test, y_pred)


# Karışıklık matrisi değerleri
from sklearn import metrics
metrics.confusion_matrix(y_test,y_pred)


deneme1=[[5.1,3.5,1.4,0.2]]
sonuc1=clf2.predict(deneme1)
print(sonuc1)


deneme2=[[7.0,3.2,4.7,1.4]]
sonuc2=clf2.predict(deneme2)
print(sonuc2)

deneme3=[[6.5,3.0,5.8,2.2]]
sonuc3=clf2.predict(deneme3)
print(sonuc3)
