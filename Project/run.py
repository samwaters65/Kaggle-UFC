# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:27:55 2019

@author: swaters
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:01:36 2019

@author: swaters
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

data_noNA = data.dropna(axis=0, how="any")

postLen = data_noNA.shape[0]

print(f"{round((postLen - beforeLen)/beforeLen, 4)*-100}% of records removed")


##############
# NEED CORRELATION REVIEW TO PRE-REMOVE VARIABLES. Then go through NA removal again (hopefully MORE records stay)


def featureSelect():
    return RandomForestRegressor(n_estimators=100)


test_size = 0.2


def findVars(df, target="WinnerTarget", importance_threshold=0.09):
    x_cols = list(df)
    x_cols.remove(target)
    for i in range(len(x_cols) + 1):
        X = df[x_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0
        )
        model = featureSelect()
        model.fit(X_train, y_train)
        lst = list(model.feature_importances_)
        minNum = min(lst)
        if minNum > importance_threshold:
            return x_cols
        to_delete = lst.index(minNum)
        if minNum < importance_threshold:
            del lst[to_delete]
            print(f"Removed {x_cols[to_delete]}")
            del x_cols[to_delete]
    # return X, y, X_train, X_test, y_train, y_test
    return x_cols


def scaleData(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.values)
    scaled_data = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    return scaled_data


scaled_data = scaleData(data_noNA)

first_iter_x_cols = findVars(scaled_data, importance_threshold=0.005)


data_reduced = data[first_iter_x_cols]
data_reduced.insert(
    loc=data_reduced.shape[1], column="WinnerTarget", value=data.WinnerTarget
)

beforeLen = data_reduced.shape[0]

data_reduced_noNA = data_reduced.dropna(axis=0, how="any")

postLen = data_reduced_noNA.shape[0]

print(f"{round((postLen - beforeLen)/beforeLen, 4)*-100}% of records removed")


df = data_reduced_noNA


second_iter_x_cols = findVars(data_reduced_noNA, importance_threshold=0.01)


data_reduced = data[second_iter_x_cols]
data_reduced.insert(
    loc=data_reduced.shape[1], column="WinnerTarget", value=data.WinnerTarget
)


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
