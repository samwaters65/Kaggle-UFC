# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:28:35 2019

@author: swaters
"""

shouldScale = True


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
