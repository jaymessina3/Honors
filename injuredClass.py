#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 23:47:36 2019

@author: JayMessina
"""

import pandas
import os

def main():
    names = ["Date", "Team", "Acquired", "Relinquished", "Notes", "extra"]
    dataset = pandas.read_excel("2013-2014_injured_edits.xlsx", names=names)
    array = dataset.values
    d = {}
    print("here")
    for x in array:
        date = str(x[0])
        name = x[3]
        injury = str(x[4])
        if(type(name)==float): 
            continue
        name = name.split(' ', 1)[1]
        date = date.split(' ')[0]
        c = {name: {date:injury}}
        for k in d.keys():
             if name==k:
                 #print(d[name])
                 c[name].update(d[name])
                 break
                 #c = {name: {d[name]}, {{date:injury}}}
                # d[name].append({date:injury})
             
                
        d.update(c)
    #print(len(d))
    #print(d)
    #for i in d:
        #print(i)
    directory = os.fsencode("2014")
    print(directory)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".xlsx"): 
            # print(os.path.join(directory, filename))
            continue
        else:
            continue
        
main()