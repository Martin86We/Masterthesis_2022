#!/usr/bin/env python
# coding: utf-8

# ## Lineare Regression
# 
# #### Beispiel: Kilometer in Meilen umrechnen

# Ein einzelnes Neuron kann den Zusammenhang zwischen Kilometer und Meilen "lernen".

# In[1]:


X = [
    [10],
    [15],
    [60]
]

y = [
    6.2,
    9.3,
    37.3
]


# In[2]:


from sklearn.linear_model import LinearRegression       

model = LinearRegression(fit_intercept = False)         
model.fit(X, y)                                         


# In[3]:


model.coef_                                             


# In[4]:


print(120 * 0.62152866)                               


# In[5]:


model.predict([
    [120],
    [130]
])
                                                        


# In[ ]:




