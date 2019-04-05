
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

xNate = []
yNate = []

totalCounter = 0
majorCounter = 0

'''
d1 = "2013-12-25"
d2 = "2014-1-01"
d1 = datetime.strptime(d1, "%Y-%m-%d")
d2 = datetime.strptime(d2, "%Y-%m-%d")
d1 = (d1 + timedelta(days=14))
print(str(d1.date()))
ds = str(d1.date())
dx = ds.split(" ")[0]
#d1 = datetime.strptime(d1, "%Y-%m-%d")
print(dx)
print( abs((d2 - d1).days))

'''

def loadData1(directory):
    names = ['GM', 'Date', 'Weight', 'MP', 'FGA', '3PA', 'FTA', 'ORB', 'DRB', 'AST', 'TO', 'Fouls', 'PTS', 'dist_feet', 'avg_speed', "post_ups", "drives", "Injury"]
    
    global xFinal
    global yFinal
    
    global nateX
    global nateY
    
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
        counter = 0
        
        '''
        date = array[:,1:2]
        startAgg = datetime.strptime(date[0], "%Y-%m-%d")
        endAgg = (startAgg + timedelta(days=14))
        startPre = datetime.strptime(date[14], "%Y-%m-%d")
        endPre = (startPre + timedelta(days=7))
        c = 0
        while(str(date[c][0])!="nan"):
            xNew = []
        
        '''
            
            
        flag = 0
        
        countGames = -1

        
        
        
        for i in range(0, len(X)-aWind-pWind):
            xNew = []
            flag = 0
            addition = 0
            for k in range(0+i,aWind+i):
                if(str((X[k][1]))=="nan"):
                    continue
                played = X[k][1]
                xNew.extend(X[k])
                if(played==0):
                    xNew.extend([0])
                else:
                    if(flag == 0):
                        countGames+=1
                        flag = 1
                    xNew.extend([1])
                    addition+=1
                if(str(game[k][0])!="nan"):
                    xNew.extend(game[k])                
                else:
                    xNew.extend([0])
                #xNew.extend([countGames+addition])
                
                
            if(filename == "Nate Robinson_2014.xlsx"):
                xNate.append(xNew)
            else:
                xFinal.append(xNew)
        
        '''
        for line in xFinal:
            print(line)
            print()
        break
        '''
        for i in range(aWind, len(Y)-pWind):
            yNew = []
            for k in range(0+i,pWind+i):
                played = X[k][1]
                if(len((str(Y[k])))<2 or (str(Y[k])=="nan")):
                    yNew.append("F")
                else:
                    #print(Y[k])
                    yNew.append("T")
            #print(yNew)
            
            counter = 0
            for injury in yNew:
                if injury != "F":
                    counter+=1
            if(counter>=3):
                majorCounter+=1
                if(filename == "Nate Robinson_2014.xlsx"):
                    yNate.append("T")
                else:
                    yFinal.append("T")

            #if(counter>0):
                #yFinal.append("T")
                totalCounter+=1
                


            else:
                if(filename == "Nate Robinson_2014.xlsx"):
                    yNate.append("F")
                else:
                    yFinal.append("F")
        
        
                
def loadData2(directory):
    names = ['GM', 'Date', 'Weight', 'MP', 'FGA', '3PA', 'FTA', 'ORB', 'DRB', 'AST', 'TO', 'Fouls', 'PTS', 'dist_feet', 'avg_speed', "post_ups", "drives", "Injury"]
    
    global xFinal2
    global yFinal2
    
    
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
        X = array[:,2:17]
        Y = array[:,17]
        game = array[:,0:1]

        countGames = -1
            
            
        flag = 0
        for i in range(0, len(X)-aWind-pWind):
            xNew = []
            flag = 0
            addition = 0
            for k in range(0+i,aWind+i):
                if(str((X[k][1]))=="nan"):
                    continue
                played = X[k][1]
                xNew.extend(X[k])
                if(played==0):
                    xNew.extend([0])
                else:
                    if(flag == 0):
                        countGames+=1
                        flag = 1
                    xNew.extend([1])
                    addition+=1
                
                if(str(game[k][0])!="nan"):
                    xNew.extend(game[k])
                else:
                    xNew.extend([0])
                
                
            
            xFinal2.append(xNew)
        '''
        for line in xFinal:
            print(line)
            print()
        '''
        
        for i in range(aWind, len(Y)-pWind):
            yNew = []
            for k in range(0+i,pWind+i):
                played = X[k][1]
                if(len((str(Y[k])))<2 or (str(Y[k])=="nan")):
                    yNew.append("F")
                else:
                    #print(Y[k])
                    yNew.append("T")
            #print(yNew)
            
            counter = 0
            for injury in yNew:
                if injury != "F":
                    counter+=1
            if(counter>=3):
                majorCounter+=1
                yFinal2.append("T")

            #if(counter>0):
                #yFinal.append("T")
                totalCounter+=1
                


            else:
                yFinal2.append("F")

def main():
    
    directory = "bball_reference_2013-2014"
    loadData1(directory)
    directory = "bball_reference_2014-2015"
    loadData2(directory)
    
    
    print("Major injuries", majorCounter)
    print("total injuries: ", totalCounter)
    validation_size = 0.20
    seed = random.randint(1, 10000)
    
    global xFinal
    global yFinal
    global xFinal2
    global yFinal2
    
    xFinal.extend(xFinal2)
    yFinal.extend(yFinal2)
    
    print(len(xFinal)) 
    print()
    print(xFinal[len(xFinal)-1])
    print(yFinal[len(xFinal)-1]) 
    print()
    print(len(yFinal))
    
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
    
    xFalseLess = []
    yFalseLess = []
    randos = []
    rand = random.randint(0, len(xFalse)-1)
    randos.append(rand)
    
    print("Lengths True")
    print(len(xTrue))
    print(len(yTrue))
    
    
    print("Lengths False")
    print(len(xFalse))
    print(len(yFalse))
    
    counter = 0
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
           
    
    
    
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(xFinal, yFinal, test_size=validation_size, random_state=seed)
    
    
    
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
    rf2 = RandomForestClassifier(n_estimators=100)
    #got 58%
    #rf = LogisticRegression()
    global xNate
    global yNate
    
   
    
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
    
    #rf2.fit(xFinalEdit, yFinalEdit)
    
    predictions2 = rf.predict(xNate) 
    y_predict_probabilities2 = rf.predict_proba(xNate)[:,1]
    print(y_predict_probabilities2)
    pred2 = []
    for i in yNate:
        if i=="T":
            countVal+=1
            pred2.append(1)
        else:
            pred2.append(0)
    print(yNate)
    print(predictions2)
    #fpr, tpr, _ = roc_curve(pred2, y_predict_probabilities2)
    #print(fpr)
    #aucurve2 = roc_auc_score(pred2, y_predict_probabilities2)
    print("Area under curve")
    #print(aucurve2)
    
    print("nate")
    
    print(accuracy_score(yNate, predictions2))
    print(confusion_matrix(yNate, predictions2))
    print(classification_report(yNate, predictions2))
    
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






