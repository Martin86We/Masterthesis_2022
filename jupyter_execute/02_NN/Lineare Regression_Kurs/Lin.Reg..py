#!/usr/bin/env python
# coding: utf-8

# # Lineare Regression (A)

# ## Lineare Regression (manuell)

# In[1]:


# # Lineare Regression (manuell)

import numpy as np
import matplotlib.pyplot as plt


def f(a, x):
    return a * x


def J(a, x, y):
    return (y - a * x) ** 2


def J_ableitung_a(a, x, y):
    return -2 * x * (y - a * x)


point = (1, 4)
lr = 0.05
a = 1
for i in range(0, 10):
    da = J_ableitung_a(a, point[0], point[1])
    a = a - lr * da
    print(a)
    print("Kosten wenn a = " + str(round(a,3)) + ": " + str(round(J(a, point[0], point[1]),3)))

xs = np.arange(-2, 2, 0.1)
ys = f(a, xs)
plt.plot(xs, ys)

plt.scatter(point[0], point[1])
plt.show()


# ## Lineare Regression (mehrere Punkte)

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


def f(a, x):
    return a * x


def J(a, x, y):
    return (y - a * x) ** 2


def J_ableitung_a(a, x, y):
    return -2 * x * (y - a * x)


point1 = (1, 4)
point2 = (1.5, 5)

lr = 0.05
a = 1
for i in range(0, 50):
    da = J_ableitung_a(a, point1[0], point1[1])
    a = a - lr * da

    da = J_ableitung_a(a, point2[0], point2[1])
    a = a - lr * da

    cost = J(a, point1[0], point1[1]) + J(a, point2[0], point2[1])
    print("Kosten wenn a = " + str(a) + ": " + str(cost))

xs = np.arange(-2, 2, 0.1)
ys = f(a, xs)
plt.plot(xs, ys)

plt.scatter(point1[0], point1[1], color="red")
plt.scatter(point2[0], point2[1], color="green")
plt.show()


# ## Lineare Regression (vektorisieren)

# In[3]:


import numpy as np
import matplotlib.pyplot as plt


def f(a, x):
    return a * x


def J(a, x, y):
    return np.mean((y - a * x) ** 2)


def J_ableitung_a(a, x, y):
    return np.mean(-2 * x * (y - a * x))


points = np.array([
    [1, 4],
    [1.5, 5],
    [2, 8],
    [0.5, 3]
])

lr = 0.05
a = 1
for i in range(0, 50):
    da = J_ableitung_a(a, points[:, 0], points[:, 1])
    a = a - lr * da

    cost = J(a, points[:, 0], points[:, 1])
    print("Kosten wenn a = " + str(a) + ": " + str(cost))

xs = np.arange(-2, 2, 0.1)
ys = f(a, xs)
plt.plot(xs, ys)

plt.scatter(points[:, 0], points[:, 1], color="red")
plt.show()


# ## Gradientenabstieg

# In[ ]:





# In[4]:


import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 2 - 4 * x + 5


def f_ableitung(x):
    return 2 * x - 4


x = 5
lr = 0.05

plt.scatter(x, f(x), c="r")
for i in range(0, 25):
    steigung_x = f_ableitung(x)
    x = x - lr * steigung_x
    plt.scatter(x, f(x), c="r")
    print(x)

xs = np.arange(-2, 6, 0.1)
ys = f(xs)
plt.plot(xs, ys)
plt.show()


# # Lineare Regression (B)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




