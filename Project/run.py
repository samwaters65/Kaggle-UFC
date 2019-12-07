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
data = pd.get_dummies(data, columns=["weight_class", "B_Stance", "R_Stance"])

data.drop(axis=1, columns=c.unusedCols, inplace=True)
beforeLen = data.shape[0]
data_noNA = data.dropna(axis=0, how="any")
postLen = data_noNA.shape[0]

print(f"{round((postLen - beforeLen)/beforeLen, 4)*-100}% of records removed")

#### Baseline Performance (Kitchen Sink)

model, baseline_test_score = f.runLogReg(data_noNA) 

##############
# NEED CORRELATION REVIEW TO PRE-REMOVE VARIABLES. Then go through NA removal again (hopefully MORE records stay) 

 
# Scaled Model:

scaled_data = f.scaleXData(data_noNA) 
scaled_data.insert(loc=scaled_data.shape[1], column='WinnerTarget', value=data_noNA.WinnerTarget)

scaledModel, scaled_test_score = f.runLogReg(scaled_data)


# Feature Reduction Iteration 1 (low threshold)

first_iter_x_cols = f.findVars(scaled_data, importance_threshold = 0.005)

data_reduced1 = data[first_iter_x_cols]
data_reduced1.insert(loc=data_reduced1.shape[1], column='WinnerTarget', value=data.WinnerTarget)

beforeLen = data_reduced1.shape[0]
data_reduced1_noNA = data_reduced1.dropna(axis=0, how="any")
postLen = data_reduced1_noNA.shape[0]
print(f"{round((postLen - beforeLen)/beforeLen, 4)*-100}% of records removed")

data_reduced1_scaled = f.scaleXData(data_reduced1)
data_reduced1_scaled.insert(loc=data_reduced1_scaled.shape[1], column='WinnerTarget', value=data_reduced1_noNA.WinnerTarget)

firstIterModel, firstIter_test_score = f.runLogReg(data_reduced1_scaled)


# Second Iteration - Higher importance threshold
second_iter_x_cols = f.findVars(data_reduced1_noNA, importance_threshold=0.01)


data_reduced = data[second_iter_x_cols]
data_reduced.insert(loc=data_reduced.shape[1], column='WinnerTarget', value=data.WinnerTarget)


beforeLen = data_reduced.shape[0]

data_reduced_noNA = data_reduced.dropna(axis=0, how="any")

postLen = data_reduced_noNA.shape[0]

print(f"{round((postLen - beforeLen)/beforeLen, 4)*-100}% of records removed")


scaled_data = scaleData(data_reduced_noNA)


#####################
# Remove all of the unnecessary variables and then run again through findVars(). Then, remove the vars
# from FULL data and run through process again.




X = scaled_data.drop(axis=1, columns=["WinnerTarget"])
y = scaled_data.WinnerTarget

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=77
)

clf = LogisticRegression(
    penalty="elasticnet",
    random_state=64,
    solver="saga",
    max_iter=100
    # multi_class="auto",
).fit(X_train, y_train)

y_preds = clf.predict(X_test)

print(f"Base Score: {round(clf.score(X_test, y_test),4)*100}%")

## Feature selection, balancing?, scaling


### Need to use random forest classifier for
### Idea: Test if Referee is statistically biased!
