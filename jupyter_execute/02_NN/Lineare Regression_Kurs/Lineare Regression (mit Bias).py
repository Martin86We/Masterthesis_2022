#!/usr/bin/env python
# coding: utf-8

# ## Lineare Regression (mit Bias)
# 
# #### Beispiel: Grad Celsius -> Fahrenheit

# In[1]:


X = [
    [-10],
    [0],
    [20]
]

y = [
    14,
    32,
    68
]


# In[2]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)


# In[3]:


print(model.coef_)
print(model.intercept_)


# In[4]:


# X1 * 1.8 + 32

