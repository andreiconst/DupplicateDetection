#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 19:17:11 2019

@author: andrei
"""


import sys
sys.path.append("/usr/local/lib/python3.6/dist-packages")

import chars2vec
import sklearn.decomposition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# Load Inutition Engineering pretrained model
# Models names: 'eng_50', 'eng_100', 'eng_150' 'eng_200', 'eng_300'
c2v_model = chars2vec.load_model('eng_50')

#dataset
personalDetails = pd.read_csv('personalDetailsMillion.csv')


def ExtractFirstName(s):
    s = s.replace(" ", "")
    x = re.split('[^a-zA-Z0-9]', s)
    x = [x[i] for i in range(len(x)) if x[i] != '' ]
    try:
        if(x[x.index('FirstName') + 4 ] == '0'):
            return x[x.index('FirstName') + 2 ].lower()
    except:
        pass  
    
def ExtractLastName(s):
    s = s.replace(" ", "")
    x = re.split('[^a-zA-Z0-9]', s)
    x = [x[i] for i in range(len(x)) if x[i] != '' ]
    try:
        if(x[x.index('LastName') + 4 ] == '0'):
            return x[x.index('LastName') + 2 ].lower()
    except:
        pass

class PersonDetails:
  def __init__(self, Id, first_name, last_name):
    self.Id = Id
    self.first_name = first_name
    self.last_name = last_name    

    

personDetails = list(personalDetails['PersonDetails'])
personDetailList = list()
for i in range(len(personDetails)):
    personDetailList.append(PersonDetails(i, ExtractFirstName(personDetails[i]), ExtractLastName(personDetails[i])))


del(personDetails)
del(personalDetails)

lastNamesCleaned = list()
indexCorrespondance = list()
for i, pdt in enumerate(personDetailList):
    if pdt.last_name != None:
        indexCorrespondance.append(i)
        lastNamesCleaned.append(pdt.last_name)

#Step 1 : transform word into vectors
word_embeddings = c2v_model.vectorize_words(lastNamesCleaned)

#Step2 Normal Hash Function

def HashFunction(matrix):
    return matrix > 0

def NormalizeVectors(matrix):
    return (matrix.T / (np.sqrt(np.diag(np.dot(matrix, matrix.T))))).T


def augmentDictionary(dictionaryTemp, dictionaryFinal):
    for k in dictionaryTemp.keys():
        if (len(dictionaryTemp[k]) > 1):
            dictionaryFinal[k] = dictionaryTemp[k]
    
    
def DispatchInDictionary(wordEmbeddings):
    dictionary = dict()
    for i in range(50):
        dictTemp = dict()
        random_vectors = np.random.multivariate_normal(np.zeros(50), np.identity(50), 20)
        random_projection = np.dot(wordEmbeddings, random_vectors.T)
        word_embeddings_hashed = HashFunction(random_projection)
        for j in range(random_projection.shape[0]):
            signature = str(word_embeddings_hashed[j,:])
            try:
                dictTemp[str(i) + signature].append(j)
            except:
                dictTemp[str(i) + signature] = list()
                dictTemp[str(i) + signature].append(j)
        augmentDictionary(dictTemp, dictionary)
        print(i)
    return dictionary

def DoComparisons(signatureDictionary, distanceMatrix, threshold = 0.95):
    alreadyCompared = set()
    similarPairs = list()
    #dictionaryPersonIdToGroupId = dict()
    #dictionaryGroupIdToPersonId = dict()

    #groupId = 0
    for cc, k in enumerate(signatureDictionary.keys()):
        if len(signatureDictionary[k]) > 1:
            for j in range(len(signatureDictionary[k])-1):
                index1 = signatureDictionary[k][j]
                index2 = signatureDictionary[k][j+1]
                comparison = str(index1) + ' vs ' + str(index2)
                if comparison not in alreadyCompared :
                    alreadyCompared.add(comparison)
                    value = np.dot(distanceMatrix[index1,:], distanceMatrix[index2,:].T) / (np.linalg.norm(distanceMatrix[index1,:]) * np.linalg.norm(distanceMatrix[index2,:]))
                    if value > threshold:
                        similarPairs.append([index1, index2])
                        #groupId = AggregateGroup(index1, index2, dictionaryPersonIdToGroupId, dictionaryGroupIdToPersonId, groupId)
        if (cc % 10000 == 0):
            print(cc / len(signatureDictionary.keys()))
    print(len(alreadyCompared))
    return similarPairs

def AggregateGroup(listEqualPairs):
    countSofar = 0
    dictionary1 = dict()
    dictionary2 = dict()
    for kk,listEqualPair in enumerate(listEqualPairs):
        index1 = listEqualPair[0]
        index2 = listEqualPair[1]
        if (index1 in dictionary1.keys()) and (index2 in dictionary1.keys()) and (dictionary1[index1] != dictionary1[index2]): # both already have groups, fucked
            copyListGroupId = dictionary2[dictionary1[index2]][:]
            del dictionary2[dictionary1[index2]]
            for k in copyListGroupId:
                dictionary1[k] = dictionary1[index1]
                dictionary2[dictionary1[index1]].append(k)
    
        elif (index1 in dictionary1.keys()): #easy add second element group id first element, update both dictionaries
            dictionary1[index2] = dictionary1[index1]
            dictionary2[dictionary1[index1]].append(index2)
        elif (index2 in dictionary1.keys()): #easy add first element group id second element, update both dictionaries
            dictionary1[index1] = dictionary1[index2]
            dictionary2[dictionary1[index2]].append(index1)
        else:
            dictionary1[index1] = countSofar
            dictionary1[index2] = countSofar
            dictionary2[countSofar] = list()
            dictionary2[countSofar].append(index1)
            dictionary2[countSofar].append(index2)
            countSofar += 1
        if (kk % 10000 == 0):
            print(kk / len(listEqualPairs))
    return dictionary1, dictionary2

def DictionaryToReadable(dictionary):
    readableDict = dict()
    for k in dictionary.keys():
        temp = list()
        for j in range(len(dictionary[k])):
            temp.append(lastNamesCleaned[dictionary[k][j]])
        readableDict[k] = temp
    return readableDict
        

dictionary = DispatchInDictionary(word_embeddings)
result = DoComparisons(dictionary, word_embeddings)
dictionaryPersonIdToGroupId, dictionaryGroupIdToPersonId = AggregateGroup(result)
dictionaryGroupIdToPersonIdReadable = DictionaryToReadable(dictionaryGroupIdToPersonId)

