#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 10:25:35 2018

@author: JayMessina
"""

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

names = ["Date", "Team", "Acquired", "Relinquished", "Notes"]
dataset = pandas.read_excel("2013-2014_injured_edits.xlsx", names=names)
array = dataset.values
#print(array)

d = {''}

for x in array:
    s = x[3]
    if(type(s)==float): 
        continue
    s = s.split(' ', 1)[1]
    d.add(s)
file = open("injured_list.txt", "w")
for x in d:
    file.write('%s\n' % (x))
    print(x)
print (d)
print(len(d))

#look for these names in file and then for each name in big file, add T or F if injured

