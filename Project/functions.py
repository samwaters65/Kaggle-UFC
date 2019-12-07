# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:28:21 2019

@author: swaters
"""
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import config as c
import models as m

def isUSA(location):
    if "USA" in location:
        return 1
    else:
        return 0


def winner(winner):
    if winner == "Blue":
        return 1
    if winner == "Red":
        return 2
    if winner == "Draw":
        return 0


def findVars(df, target='WinnerTarget', importance_threshold=0.09):
    x_cols = list(df)
    x_cols.remove(target)
    for i in range(len(x_cols) + 1):
        X = df[x_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = c.test_size, random_state = 0)
        model = m.featureSelect()
        model.fit(X_train, y_train)
        lst = list(model.feature_importances_)
        minNum = min(lst)
        if minNum > importance_threshold:
            return x_cols
        to_delete = lst.index(minNum)
        if minNum < importance_threshold:
            del lst[to_delete]
            print(f'Removed {x_cols[to_delete]}')
            del x_cols[to_delete]
    #return X, y, X_train, X_test, y_train, y_test
    return x_cols

def scaleData(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.values)
    scaled_data = pd.DataFrame(scaled_features, index = df.index, columns = df.columns)
    return scaled_data
