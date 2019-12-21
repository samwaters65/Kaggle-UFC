# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:27:55 2019

@author: swaters
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import functions as f
import config as c

data = pd.read_csv(f"{c.filesDir}data.csv")

# New Features
data["USAFight"] = data.location.apply(f.isUSA)
data["WinnerTarget"] = data.Winner.apply(f.winner)
data = pd.get_dummies(data, columns=["weight_class",
                                         "B_Stance",
                                         "R_Stance"])

data.drop(axis=1, columns=c.unusedCols, inplace=True)
beforeLen = data.shape[0]
data_noNA = data.dropna(axis=0, how="any")
postLen = data_noNA.shape[0]

print(f"{round((postLen - beforeLen)/beforeLen, 4)*-100}% of records removed")

#### Baseline Performance (Kitchen Sink)

model, baseline_test_score = f.runLogReg(data_noNA) 

 
# Scaled Model:

scaled_data = f.scaleXData(data_noNA) 
scaled_data.insert(loc=scaled_data.shape[1],
                   column='WinnerTarget',
                   value=data_noNA.WinnerTarget)

scaledModel, scaled_test_score = f.runLogReg(scaled_data)


# Feature Reduction Iteration 1 (low threshold)

first_iter_x_cols = f.findVars(scaled_data, importance_threshold = 0.005)
print(f'Number of X_vars: {len(first_iter_x_cols)}')

data_reduced1 = data[first_iter_x_cols]
data_reduced1.insert(loc=data_reduced1.shape[1],
                     column='WinnerTarget',
                     value=data.WinnerTarget)

beforeLen = data_reduced1.shape[0]
data_reduced1_noNA = data_reduced1.dropna(axis=0, how="any")
postLen = data_reduced1_noNA.shape[0]
print(f"{round((postLen - beforeLen)/beforeLen, 4)*-100}% of records removed")

data_reduced1_scaled = f.scaleXData(data_reduced1_noNA)
data_reduced1_scaled.insert(loc=data_reduced1_scaled.shape[1],
                            column='WinnerTarget',
                            value=data_reduced1_noNA.WinnerTarget)

firstIterModel, firstIter_test_score = f.runLogReg(data_reduced1_scaled)


# Second Iteration - Higher importance threshold
second_iter_x_cols = f.findVars(data_reduced1_noNA, importance_threshold=0.015)
print(f'Number of X_vars: {len(second_iter_x_cols)}')

data_reduced = data[second_iter_x_cols]
data_reduced.insert(loc=data_reduced.shape[1],
                    column='WinnerTarget',
                    value=data.WinnerTarget)


beforeLen = data_reduced.shape[0]
data_reduced2_noNA = data_reduced.dropna(axis=0, how="any")
postLen = data_reduced2_noNA.shape[0]

print(f"{round((postLen - beforeLen)/beforeLen, 4)*-100}% of records removed")


data_reduced2_scaled = f.scaleXData(data_reduced2_noNA)
data_reduced2_scaled.insert(loc=data_reduced2_scaled.shape[1],
                            column='WinnerTarget',
                            value=data_reduced2_noNA.WinnerTarget)
secondIterModel, secondIter_test_score = f.runLogReg(data_reduced2_scaled)



# Third Iteration
third_iter_x_cols = f.findVars(data_reduced2_noNA, importance_threshold=0.15)
print(f'Number of X_vars: {len(third_iter_x_cols)}')

data_reduced = data[third_iter_x_cols]
data_reduced.insert(loc=data_reduced.shape[1],
                    column='WinnerTarget',
                    value=data.WinnerTarget)


beforeLen = data_reduced.shape[0]
data_reduced3_noNA = data_reduced.dropna(axis=0, how="any")
postLen = data_reduced3_noNA.shape[0]

print(f"{round((postLen - beforeLen)/beforeLen, 4)*-100}% of records removed")


data_reduced3_scaled = f.scaleXData(data_reduced3_noNA)
data_reduced3_scaled.insert(loc=data_reduced3_scaled.shape[1],
                            column='WinnerTarget',
                            value=data_reduced3_noNA.WinnerTarget)
thirdIterModel, thirdIter_test_score = f.runLogReg(data_reduced3_scaled)
