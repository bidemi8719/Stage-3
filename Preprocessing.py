# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:52:41 2020

@author: TEMITAYO
"""

import tensorflow as tf
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras import optimizers
import keras.backend
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split

print(keras.backend.backend())

#os.getwd()


DATAdir = 'C:/Users/TEMITAYO/Pictures/All_chili_data'
os.chdir("C:\\Users\\TEMITAYO\\Desktop\\CS 519 Asn")


CATEGORIES = ["disease", "normal"]
img_size = 150
training_data = []

def create_training_data():
    for category in CATEGORIES :
        path = os.path.join(DATAdir, category) # path to normal and disease
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                #img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
create_training_data()


print(len(training_data))


for sample in training_data[:10]:
    print(sample[1])
    
    
x = []
y = []
for features, lable in training_data:
    x.append(features)
    y.append(lable)
X = np.array(x).reshape(-1, img_size, img_size, 3)


X = X/255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#saving whAT we have done
pickle_out = open("X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()
pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()
pickle_out = open(" y_train.pickle", "wb")
pickle.dump( y_train, pickle_out)
pickle_out.close()
pickle_out = open("y_test.pickle", "wb")
pickle.dump( y_test, pickle_out)
pickle_out.close()












