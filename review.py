# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:01:36 2019

@author: swaters
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pandas as pd


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


filesDir = """C:\\Users\\swaters.PMIDSD\\Documents\\Continual Learning\\
ML Practice\\Kaggle\\UFCData\\"""

data = pd.read_csv(f"{filesDir}data.csv")

# New Feature
data["USAFight"] = data.location.apply(isUSA)
data["WinnerTarget"] = data.Winner.apply(winner)


data = pd.get_dummies(data, columns=["weight_class", "B_Stance", "R_Stance"])

unusedCols = [
    "R_fighter",
    "B_fighter",
    "date",
    "location",
    "B_draw",
    "R_draw",
    "Winner",
    "Referee",
]

data.drop(axis=1, columns=unusedCols, inplace=True)

beforeLen = data.shape[0]

data.dropna(axis=0, how="any", inplace=True)

postLen = data.shape[0]

print(f"{round((postLen - beforeLen)/beforeLen, 4)*-100}% of records removed")


X = data.drop(axis=1, columns=["WinnerTarget"])
y = data.WinnerTarget

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=77
)

clf = LogisticRegression(
    penalty="elasticnet", random_state=64, solver="saga", max_iter=100
).fit(X_train, y_train)


print(f"Base Score: {round(clf.score(X_test, y_test),4)*100}%")


### Need to use random forest classifier for
### Idea: Test if Referee is statistically biased!
