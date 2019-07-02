#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 19:59:15 2019

@author: andrei
"""

#Boxes and prisonners

import numpy as np
results = list()
import random
totalRounds = 10000


def checkCycle(box, ind, boxToCheck, setToCheck):
    nbInBox = -1
    count = 0

    while(nbInBox != ind):
        if (count== 0):
            toOpen = ind
        else:
            toOpen = nbInBox
            
        count += 1
        nbInBox = boxToCheck[toOpen]
        setToCheck.remove(toOpen)
    
    return count


count = 0
for i in range(totalRounds):
    boxesToCheck = np.array([i for i in range(100)])
    random.shuffle(boxesToCheck)
    boxesSet = set([i for i in range(100)])

    
    while (len(boxesSet) > 0):
        index = list(boxesSet)[0]
        lenCycle = checkCycle(boxes, index, boxesToCheck, boxesSet)
        if (lenCycle > 50):
            break
        if len(boxesSet) == 0:
            count += 1


print(count / totalRounds)

np.log(2)
    