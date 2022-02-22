#!/usr/bin/env python
# coding: utf-8

# # NN 2 (Mehrere unterscheiden)

# In[1]:


import gzip
import numpy as np
from tensorflow.keras.utils import to_categorical
from numpy import load
import matplotlib.pyplot as plt


# ## Datensatz laden

# In[2]:


X_train = load('Dataset/X_train.npy').astype(np.float32).reshape(-1, 784)*1.0/255.0
y_train = load('Dataset/y_train.npy').astype(np.int32)

X_test=load('Dataset/X_test.npy').astype(np.float32).reshape(-1, 784)*1.0/255.0
y_test=load('Dataset/y_test.npy').astype(np.int32)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[3]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ## Modell erstellen

# In[4]:


model = Sequential()

model.add(Dense(50, activation="sigmoid", input_shape=(784,)))
model.add(Dense(5, activation="sigmoid"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])


# In[5]:


X_train.reshape(10500, 784)


# ## Modell trainieren

# In[6]:


model.fit(
    X_train.reshape(10500, 784),
    y_train,
    epochs=80,
    batch_size=500)


# ## Evaluieren

# In[7]:


model.evaluate(X_test.reshape(-1, 784), y_test)


# In[8]:


model.predict(X_test.reshape(-1, 784))


# ## Model testen

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

print(y_test[0])

plt.imshow(X_test[0].reshape(28,28), cmap="gray")
plt.show()


# In[10]:


pred = model.predict(X_test.reshape(-1, 784))


# In[11]:


import numpy as np
# Klasse mit höchster Wahrscheinlichkeit ausgeben:
np.argmax(pred[0])


# **Das Modell hat Klasse 4 korrekt erkannt.**

# In[12]:


np.argmax(pred, axis=1)


# ## Confusion Matrix

# In[13]:


import pandas as pd
ytrue = pd.Series(np.argmax(y_test, axis= 1), name = 'ytrue')
ypred = pd.Series(np.argmax(pred, axis= 1), name = 'pred')
pd.crosstab(ytrue, ypred)

