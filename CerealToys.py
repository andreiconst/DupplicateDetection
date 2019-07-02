#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 19:45:11 2019

@author: andrei
"""

## Cereal toys

import numpy as np

results  = list()

for i in range(10000):
    toysToGet = set([1,2,3,4])
    count = 0
    while(len(toysToGet) > 0):
        cerealToy = np.random.randint(1,5)
        count+=1
        try:
            toysToGet.remove(cerealToy)
        except:
            pass
        if (len(toysToGet) == 0):
            results.append(count)
        
print(np.mean(results))


