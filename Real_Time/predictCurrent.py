
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 00:03:31 2018

@author: JayMessina
"""
#leaguedashplayerstats
import random 
import math
# Load libraries
import pandas
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import roc_auc_score, auc, roc_curve
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
import openpyxl
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from datetime import datetime
from datetime import timedelta 


xFinal = []
yFinal = []

xFinal2 = []
yFinal2 = []

xPlayer = []
yPlayer = []

totalCounter = 0
majorCounter = 0

player = ""

#"LeBron James_2019.xlsx"


def loadData1(directory):
    global player
    names = ['GM', 'Date', 'Weight', 'MP', 'FGA', '3PA', 'FTA', 'ORB', 'DRB', 'AST', 'TO', 'Fouls', 'PTS', 'dist_feet', 'avg_speed', "post_ups", "drives", "Injury"]
    
    '''
    global xFinal
    global yFinal
    '''
    xFinal = []
    yFinal = []
    
    global xPlayer
    global yPlayer
    
    aWind = 8
    pWind = 4
    
    global totalCounter 
    global majorCounter
    for filename in os.listdir(directory):
        if(filename==".DS_Store"):
            continue
        print(filename)
        file = directory + "/" + filename
        dataset = pandas.read_excel(file, names=names)
        array = dataset.values
        #game = array[:,0,1]
        game = array[:,0:1]
        
        X = array[:,2:17]
        Y = array[:,17]
        
        
        
        flag = 0
        for i in range(0, len(X)-aWind-pWind):
            xNew = []
            yNew = []
            flag = 0
            for k in range(i, aWind+i):
                played = X[k][1]
                    
                if(played == 0 and (len((str(Y[k])))<2 or (str(Y[k])=="nan"))):
                    flag = 1
                    break
                
                elif(played==0): #and were injured
                    
                    xNew.extend([0])
                else:
                    xNew.extend([1])
                
                xNew.extend(X[k])
                
                #xNew.extend(game[k])
            if(flag == 1):
                continue
            
            if(filename == player):
                xPlayer.append(xNew)
            else:
                xFinal.append(xNew)
            
            for j in range(i+aWind, i+aWind+pWind):
                if((len((str(Y[k])))<2 or (str(Y[k])=="nan"))):
                    yNew.append("F")
                else:
                    yNew.append("T")
            
            counter = 0
            for injury in yNew:
                if injury != "F":
                    counter+=1
            if(counter>=1):
                if(filename == player):
                    yPlayer.append("T")
                else:
                    yFinal.append("T")
            else:
                if(filename == player):
                    yPlayer.append("F")
                else:
                    yFinal.append("F")
    return xFinal, yFinal
        
                
def loadPlayer(directory):
    global player
    names = ['GM', 'Date', 'Weight', 'MP', 'FGA', '3PA', 'FTA', 'ORB', 'DRB', 'AST', 'TO', 'Fouls', 'PTS', 'dist_feet', 'avg_speed', "post_ups", "drives", "Injury"]
    
    
    global xPlayer
    global yPlayer
    
    aWind = 8
    #pWind = 4
    path = directory+ "/" + player
    
    
    dataset = pandas.read_excel(path, names=names)
    array = dataset.values
    #game = array[:,0,1]
    game = array[:,0:1]
    
    X = array[:,2:17]
    Y = array[:,17]
    
    xFinal = []
    xNew = []
    for i in range(len(X)-aWind, len(X)):
        played = X[i][1]
        if(played == 0 and (len((str(Y[i])))<2 or (str(Y[i])=="nan"))):
            print("here")
            return [[]]
        
        elif(played==0): #and were injured
            
            xNew.extend([0])
        else:
            xNew.extend([1])
        xNew.extend(X[i])
        
    xFinal.append(xNew)
    
    return xFinal

def main():
    
    global player
    name = "Tyus Jones"
    player = name + "_2019.xlsx"
    
    
    directory = "bball_reference_2013-2014"
    xFinal, yFinal = loadData1(directory)
    directory = "bball_reference_2014-2015"
    xFinal2, yFinal2 = loadData1(directory)
    
    
    directory = "bball_reference_2018-2019"
    xFinal3, yFinal3 = loadData1(directory)
    
    
    directory = "bball_reference_2018-2019"
    xFuture = loadPlayer(directory)
    
    print(xFuture)
    
    
    
    
    validation_size = 0.20
    seed = random.randint(1, 10000)
    
    #make xFinal and yFinal contain all data but particular player we want to look at
    xFinal.extend(xFinal2)
    xFinal.extend(xFinal3)
    
    yFinal.extend(yFinal2)
    yFinal.extend(yFinal3)
    
    #used to solve class imbalance problem. with these arrays we have the same amount of true and false values
    xTrue = []
    xFalse = []
    yTrue = []
    yFalse = []
    
    
    for i in range(0, len(xFinal)):
        if(yFinal[i]=="T"):
            xTrue.append(xFinal[i])
            yTrue.append(yFinal[i])
            
        else:
            xFalse.append(xFinal[i])
            yFalse.append(yFinal[i])
    

    
    randos = []
    rand = random.randint(0, len(xFalse)-1)
    randos.append(rand)
    
    counter = 0
    
    #these array have the same amount of true and false values
    xFinalEdit = []
    yFinalEdit = []
    
    for i in range(0,len(xTrue)):
        
        while(rand in randos):
            rand = random.randint(0, len(xFalse)-1)
        randos.append(rand)
        
        xFinalEdit.append(xFalse[rand])
        yFinalEdit.append(yFalse[rand])
        
        
        xFinalEdit.append(xTrue[counter])
        yFinalEdit.append(yTrue[counter])
        counter+=1
           
    
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(xFinalEdit, yFinalEdit, test_size=validation_size, random_state=seed)
    
    scoring = 'accuracy'
    
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    #models.append(('SVM', SVC()))
    #n_estimators is number of trees in forest
    models.append(("RF", RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)))
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
    
    
    
    #got 90%
    rf = RandomForestClassifier(n_estimators=100)
    #got 58%
    #rf = LogisticRegression()
    
    
    rf.fit(X_train, Y_train)
    
    predictions = rf.predict(X_validation) 
    
    y_predict_probabilities = rf.predict_proba(X_validation)[:,1]
    #print(y_predict_probabilities)
    
    countVal = 0
    pred1 = []
    for i in Y_validation:
        if i=="T":
            countVal+=1
            pred1.append(1)
        else:
            pred1.append(0)
            
    countPred = 0
    pred2 = []
    for i in predictions:
        if i=="T":
            countPred+=1
        
    
    #print(predictions)
    #use R for data analysis
    #google class imbalance
    #training set, only have 1000 falses, pick randomely
    #chooses .1, .2 different thresholds
    fpr, tpr, _ = roc_curve(pred1, y_predict_probabilities)
    #print(fpr)
    aucurve = roc_auc_score(pred1, y_predict_probabilities)
    print("Area under curve")
    print(aucurve)
    
    print("True amount in validation ", countVal)
    print("True amount in pred ", countPred)
    
    print("accuracy")
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    
    
    #now look at particular player's data
    global xPlayer
    global yPlayer
    predictions2 = rf.predict(xPlayer) 
    y_predict_probabilities2 = rf.predict_proba(xPlayer)[:,1]
    
    print("Actual results")
    print(yPlayer)
    print("Predicted probabilities")
    print(y_predict_probabilities2)
    
    pred2 = []
    for i in yPlayer:
        if i=="T":
            countVal+=1
            pred2.append(1)
        else:
            pred2.append(0)
   
    print("Area under curve")
    #print(aucurve2)
        
    print(accuracy_score(yPlayer, predictions2))
    print(confusion_matrix(yPlayer, predictions2))
    print(classification_report(yPlayer, predictions2))
    
    
    print()
    ###games that haven't happened yet
    
    if xFuture == [[]]:
        print("Did not play in some of last 8 games")
    else:
        predictions3 = rf.predict(xFuture) 
        y_predict_probabilities3 = rf.predict_proba(xFuture)[:,1]
        
        print("Next 4 games predicted probabilities")
        print(predictions3)
        print(y_predict_probabilities3)
        
    
    '''
    printTrue = 0
    for i in predictions:
        if(i=="T"):
            printTrue+=1
    
    printVal = 0
    for i in Y_validation:
        if(i=="T"):
            printVal+=1
            
    print(printTrue, printVal)
    print(len(predictions), len(Y_validation))
    
    '''





main()






