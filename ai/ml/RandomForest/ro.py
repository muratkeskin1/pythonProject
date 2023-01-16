# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:14:07 2022

@author: SerkanSavas
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as randomforest
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  accuracy_score, confusion_matrix, classification_report

iris = datasets.load_iris()
irisdata= irisdata=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target})

X = irisdata [['sepal length', 'sepal width', 'petal length', 'petal width']]
y = irisdata ['species']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

num_trees = 100
max_features = 3
RF = randomforest (criterion='gini', bootstrap=True,
                                n_estimators=num_trees, max_features=max_features)

model =RF. fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
