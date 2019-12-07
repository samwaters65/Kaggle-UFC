# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:38:32 2019

@author: swaters
"""
from sklearn.ensemble import RandomForestRegressor

def featureSelect():
    return RandomForestRegressor(n_estimators=100)