#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 00:03:31 2018

@author: JayMessina
"""
#leaguedashplayerstats
import random 

# Load libraries
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
from sklearn.ensemble import RandomForestClassifier

#names = ["PLAYER_ID", "PLAYER_NAME", "GP", "MIN", "FGA","FTA", "DIST_FEET", "AVG_SPEED", "INJURED"]
names = ["PLAYER_ID", "PLAYER_NAME", "GP", "MIN", "DIST_FEET", "AVG_SPEED", "FGA", "INJURED"]
dataset = pandas.read_excel("2013-2015_combo_5.xlsx", names=names)

'''
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
'''

# shape
print(dataset.shape)
#print(dataset.groupby('GP').size())
dataset.hist()
print("hi")
#scatter_matrix(dataset)

array = dataset.values
#print(array)
X = array[:,2:7]
print("x", X)
Y = array[:,7]
print("y", Y)

'''
X = array[:,0:4]
Y = array[:,4]
print(X)
print("here")
print(Y)
'''
validation_size = 0.10
seed = random.randint(1, 10000)
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(("RF", RandomForestClassifier()))
#models.append(("For", RandomForestRegressor()))
# evaluate each model in turn
results = []
names = []

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

rf = RandomForestClassifier()
rf.fit(X, Y)

predictions = rf.predict(X_validation) 
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))