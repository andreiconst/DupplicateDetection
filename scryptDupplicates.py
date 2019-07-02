#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 19:02:07 2019

@author: andrei
"""

import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


#Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')




test = inception_model.layers[312].output
len(inception_model.layers)

from keras.models import Model

intermediate_layer_model = Model(inputs=inception_model.input,
                                 outputs=inception_model.get_layer("predictions").output)


def getFeatures(filepath):
    original = load_img(filepath, target_size=(299, 299))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    features = intermediate_layer_model.predict(image_batch)
    return features / np.sqrt(np.dot(features, features.T))

features1 = getFeatures("Data/im1Plagiat.png")
features2 = getFeatures("Data/im2Plagiat.png")

dot = np.dot(features1, features2.T)
dot = np.dot(features1, features1.T)
dot = np.dot(features2, features2.T)

import sys
import PyPDF2
from PIL import Image
with open("Data/extract1.pdf","rb") as file:
    file.seek(0)
    pdf = file.read()
