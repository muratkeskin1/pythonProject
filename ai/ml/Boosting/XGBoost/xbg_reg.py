# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:52:23 2022

@author: SerkanSavas
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RepeatedKFold
from xgboost import XGBRegressor

# veri kümesi yükleme
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = pd.read_csv(url, header=None)
data = dataframe.values

# veri kümesini düzenleme
X, y = data[:, :-1], data[:, -1]

# model tanımlama
model = XGBRegressor()

# K-fold çapraz doğrulama
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# model değerlendirme
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )