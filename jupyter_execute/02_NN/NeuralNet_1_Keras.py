#!/usr/bin/env python
# coding: utf-8

# # NN 1 ("Einer gegen Alle")

# In[1]:


import numpy as np
from numpy import load
#from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
import pickle
import matplotlib.pyplot as plt


# In diesem Kapitel wird ein Neuronales Netz erstellt, welches entscheiden kann ob ein bestimmtes Objekt auf dem Bild zu sehen ist oder etwas anderes. Dieses Netz kann also erstmal nur eine Gruppe z.B. Sechskantschraube von allen anderen unterscheiden.
# 
# Wir werden 28x28 Pixel große Bilder von Schraubenköpfen verwenden.

# In[2]:


#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ## Datensatz laden

# In[3]:


X_train = load('Dataset/X_train.npy').astype(np.float32).reshape(-1, 784)*1.0/255.0
y_train = load('Dataset/y_train.npy').astype(np.int32)

X_test=load('Dataset/X_test.npy').astype(np.float32).reshape(-1, 784)*1.0/255.0
y_test=load('Dataset/y_test.npy').astype(np.int32)

y_train_A = y_train == 3 #sechskant soll erkannt werden
y_train_B = y_train == 2 #pozidriv soll erkannt werden


# In[4]:


X_train.shape


# In[5]:


i=0
print(y_train_A[i])
#print(X_train[i].reshape(28,28))
plt.imshow(X_train[i].reshape(28,28)*255.0,cmap='gray',vmin=0,vmax=255)
plt.show
# 0: innensechskant
# 1: philips
# 2: pozidriv
# 3: sechskant
# 4: torx


# ## Neuronales Netz A (Sechskant)

# In[6]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(16, activation="sigmoid", input_shape=(784,)))
model.add(Dense(1, activation="sigmoid"))

#sgd = stochastic gradient descent
model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])


#####################
model.fit(
    X_train,
    y_train_A,
    epochs=20,
    batch_size=500)


# ## Neuronales Netz B (Pozidriv)

# In[7]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(16, activation="sigmoid", input_shape=(784,)))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

#################
model.fit(
    X_train,
    y_train_B,
    epochs=10,
    batch_size=500)


# In[8]:


model.evaluate(X_train.reshape(10500, 784), y_train)


# In[9]:


get_ipython().run_line_magic('pinfo', 'model.evaluate')


# In[10]:


print(model.metrics_names)


# In[ ]:




