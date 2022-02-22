#!/usr/bin/env python
# coding: utf-8

# # CNN1

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# In[2]:


# Vorstellung: MNIST-Daten!
# http://yann.lecun.com/exdb/mnist/
# FashionMNIST: https://github.com/zalandoresearch/fashion-mnist

import gzip
import numpy as np
from numpy import load
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


# In[3]:


X_train = load('Dataset/X_train.npy').astype(np.float32).reshape(-1, 28,28,1)
y_train = load('Dataset/y_train.npy')

X_test=load('Dataset/X_test.npy').astype(np.float32).reshape(-1,28,28,1)
y_test=load('Dataset/y_test.npy').astype(np.int32)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[4]:


print(X_train.shape)
y_train.shape


# In[5]:


y_train[0]


# In[6]:




model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=(28,28,1,)))
model.add(Conv2D(32,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(5, activation="softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=500)


# In[ ]:




