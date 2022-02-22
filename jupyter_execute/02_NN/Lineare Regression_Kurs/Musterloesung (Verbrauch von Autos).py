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


def mpg_to_l_per_100km(mpg): # Funktionsdefinition zur Umrechnung der Einheiten
    LITERS_PER_GALLON = 3.785411784
    KILOMETERS_PER_MILES = 1.609344

    return (100 * LITERS_PER_GALLON) / (KILOMETERS_PER_MILES * mpg)

print(mpg_to_l_per_100km(100))


# In[2]:


import pandas as pd # pandas um mit Dataframes zu arbeiten

df = pd.read_csv("mpg-dataset.csv") # csv einlesen


# In[15]:


X = df[["cylinders", "horsepower", "weight"]]


# In[16]:


y = mpg_to_l_per_100km(df["mpg"])


# In[17]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)


# In[18]:


print(model.coef_)
print(model.intercept_)


# In[19]:


print(model.predict([
    [8, 200, 2500]
]))

