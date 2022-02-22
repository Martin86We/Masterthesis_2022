#!/usr/bin/env python
# coding: utf-8

# ### Verbrauch von Autos vorhersagen
# 
# #### Aufgabe:
# 
# Eine Firma hat ein neues Auto angekündigt, aber noch keine Verbrauchsdaten angegeben. Kannst du den Verbrauch (in l/100km) des Autos schätzen, indem du ein Modell trainierst?
# 
# Das Auto hat:
# 
# - 8 Zylinder
# - 200PS
# - 2500kg
# 
# Lese dazu die Datei `mpg-dataset.csv` ein. Trainiere anschließend ein Modell, und sage den Verbrauch (in l/100km) dieses Autos vorher!

# In[1]:


def mpg_to_l_per_100km(mpg):
    LITERS_PER_GALLON = 3.785411784
    KILOMETERS_PER_MILES = 1.609344

    return (100 * LITERS_PER_GALLON) / (KILOMETERS_PER_MILES * mpg)

print(mpg_to_l_per_100km(100))


# In[2]:


import pandas as pd

df = pd.read_csv("mpg-dataset.csv")


# In[3]:


X = df[["cylinders", "horsepower", "weight"]]


# In[4]:


y = df["mpg"]


# In[5]:


# Aufgabe: Hier Lineare Regression trainieren


# In[6]:


y


# In[ ]:




