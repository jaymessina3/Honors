
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
    
    '''
    global xFinal
    global yFinal
    '''
    xFinal = []
    yFinal = []
    global xNate
    global yNate
    
    aWind = 8
    pWind = 4
    
    global totalCounter 
    global majorCounter
    s = set()
    for filename in os.listdir(directory):
        if(filename==".DS_Store"):
            continue
        
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
                if(X[k][0]==200 and played == 0 and i>60):
                    s.add(filename)
                    
                if(played == 0 and (len((str(Y[k])))<2 or (str(Y[k])=="nan"))):
                    flag = 1
                    break
                '''
                elif(played==0): #and were injured
                    
                    xNew.extend([0])
                else:
                    xNew.extend([1])
                '''
                
                xNew.extend(X[k])
                
                #xNew.extend(game[k])
            if(flag == 1):
                continue
            
            if(filename == "Nate Robinson_2014.xlsx"):
                xNate.append(xNew)
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
                if(filename == "Nate Robinson_2014.xlsx"):
                    yNate.append("T")
                else:
                    yFinal.append("T")
            else:
                if(filename == "Nate Robinson_2014.xlsx"):
                    yNate.append("F")
                else:
                    yFinal.append("F")
    print(s)   
    return xFinal, yFinal
        
                


def main():
    
    directory = "bball_reference_2013-2014"
    xFinal, yFinal = loadData1(directory)
    directory = "bball_reference_2014-2015"
    xFinal2, yFinal2 = loadData1(directory)
    
    print(len(xFinal))
    print(len(yFinal))
    print("Major injuries", majorCounter)
    print("total injuries: ", totalCounter)
    validation_size = 0.20
    seed = random.randint(1, 10000)
    
    '''
    global xFinal
    global yFinal
    global xFinal2
    global yFinal2
    '''
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
           
    
    #print(xFinalEdit)
    print(xFinalEdit[len(xFinalEdit)-1])
    print(len(xFinalEdit[len(xFinalEdit)-1]))
    print(len(xFinalEdit[len(xFinalEdit)-2]))
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
    '''
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    for (X_train, X_train), x in kfold:
        probas_ = RandomForestClassifier(n_estimators=100).fit(X_train, Y_train).predict_proba(X_validation)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_validation, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
    
    print(roc_auc)
    '''
    
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    
    
    
    #got 90%
    rf = RandomForestClassifier(n_estimators=100)
    #rf = LinearDiscriminantAnalysis()
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






