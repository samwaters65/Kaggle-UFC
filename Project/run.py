# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:27:55 2019

@author: swaters
"""

from sklearn.linear_model import LogisticRegression
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


##############
# NEED CORRELATION REVIEW TO PRE-REMOVE VARIABLES. Then go through NA removal again (hopefully MORE records stay) 

 

scaled_data = f.scaleData(data_noNA)
    
first_iter_x_cols = f.findVars(scaled_data, importance_threshold = 0.005)


data_reduced = data[first_iter_x_cols]
data_reduced.insert(loc=data_reduced.shape[1], column='WinnerTarget', value=data.WinnerTarget)

beforeLen = data_reduced.shape[0]

data_reduced_noNA = data_reduced.dropna(axis=0, how="any")

postLen = data_reduced_noNA.shape[0]

print(f"{round((postLen - beforeLen)/beforeLen, 4)*-100}% of records removed")




second_iter_x_cols = f.findVars(data_reduced_noNA, importance_threshold=0.01)


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
