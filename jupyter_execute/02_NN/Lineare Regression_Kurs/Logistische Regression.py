#!/usr/bin/env python
# coding: utf-8

# ## Logistische Regression
# 
# Ein einfaches Beispiel: Wird ein Studierender die Prüfung bestehen?

# In[1]:


# X = Wie viele Stunden hat er gelernt?

X = [
    [50],
    [60],
    [70],
    [20],
    [10],
    [30],
]

y = [
    1, 
    1,
    1,
    0, 
    0, 
    0,
]


# In[2]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C = 100000)
model.fit(X, y)


# In[3]:


model.predict([
    [44]
])


# In[4]:


model.predict_proba([
    [35]
])


# In[ ]:




