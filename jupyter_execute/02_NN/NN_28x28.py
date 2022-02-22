#!/usr/bin/env python
# coding: utf-8

# # Neuronales Netz (28x28)

# ## One-Hot-Label

# In[1]:


import numpy as np
from numpy import load
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
import pickle
import matplotlib.pyplot as plt


# In[2]:


# load numpy array from npy file

# load array

X_train=load('../01_Dataset/dataset_28x28/X_train.npy').astype(np.float32) * 1.0/255.0 # normalisieren
y_train=load('../01_Dataset/dataset_28x28/y_train.npy')
X_test=load('../01_Dataset/dataset_28x28/X_test.npy').astype(np.float32) * 1.0/255.0  # normalisieren
y_test=load('../01_Dataset/dataset_28x28/y_test.npy')

print(X_train.shape)
print(len(y_train))
print(X_test.shape)
print(len(y_test))


oh = OneHotEncoder()
y_train_oh = oh.fit_transform(y_train.reshape(-1, 1)).toarray()


# In[3]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:





# In[4]:


# label check
i=7
print(y_train[i])
print(y_train_oh[i])
plt.imshow(X_train[i],cmap='gray')
plt.show
# 0: innensechskant
# 1: philips
# 2: pozidriv
# 3: sechskant
# 4: torx


# In[5]:


X_train = X_train.astype(np.float32).reshape(-1, 784)#reshape hier wegen label test
X_test  = X_test.astype(np.float32).reshape(-1, 784)#
print(X_train)
print(X_test.shape)
y_test = y_test.astype(np.int)
print(y_test)


# In[6]:


class NeuralNetwork(object):
    def __init__(self, lr = 0.01):
        self.lr = lr

        self.w0 = np.random.randn(100, 784)
        self.w1 = np.random.randn(5, 100)


    def activation(self, x):
        return expit(x)

    def train(self, X, y):
        a0 = self.activation(self.w0 @ X.T)
        pred = self.activation(self.w1 @ a0)

        e1 = y.T - pred
        e0 = e1.T @ self.w1

        dw1 = e1 * pred * (1 - pred) @ a0.T / len(X)
        dw0 = e0.T * a0 * (1 - a0) @ X / len(X)

        assert dw1.shape == self.w1.shape
        assert dw0.shape == self.w0.shape

        self.w1 = self.w1 + self.lr * dw1
        self.w0 = self.w0 + self.lr * dw0

        # print("Kosten: " + str(self.cost(pred, y)))

    def predict(self, X):
        a0 = self.activation(self.w0 @ X.T)
        pred = self.activation(self.w1 @ a0)
        return pred

    def cost(self, pred, y):
        # SUM((y - pred)^2)
        s = (1 / 2) * (y.T - pred) ** 2
        return np.mean(np.sum(s, axis=0))

model = NeuralNetwork()

for i in range(0, 500):
    for j in range(0, len(X_train), 100):
        model.train(X_train[j:(j + 100), :] / 255., y_train_oh[j:(j + 100), :])

    y_test_pred = model.predict(X_test / 255.)
    y_test_pred = np.argmax(y_test_pred, axis=0)
    print(np.mean(y_test_pred == y_test))


# In[9]:


np.mean(y_test_pred == y_test)


# ## Mehrere Ausgänge

# In[10]:


import numpy as np
from tensorflow.keras.utils import to_categorical

from numpy import load
import matplotlib.pyplot as plt

X_train = load('../01_Dataset/dataset_28x28/X_train.npy').astype(np.float32).reshape(-1, 784)*1.0/255.0
y_train = load('../01_Dataset/dataset_28x28/y_train.npy').astype(np.int32)

X_test=load('../01_Dataset/dataset_28x28/X_test.npy').astype(np.float32).reshape(-1, 784)*1.0/255.0
y_test=load('../01_Dataset/dataset_28x28/y_test.npy').astype(np.int32)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[12]:


model = Sequential()

model.add(Dense(100, activation="sigmoid", input_shape=(784,)))
model.add(Dense(5, activation="sigmoid"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])


# In[15]:


model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=100)


# In[ ]:


model.evaluate(X_test.reshape(-1, 784), y_test)


# In[ ]:


model.predict(X_test.reshape(-1, 784))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

print(y_test[1])

plt.imshow(X_test[1].reshape(28,28), cmap="gray")
plt.show()


# In[ ]:


pred = model.predict(X_test.reshape(-1, 784))


# In[ ]:


import numpy as np

np.argmax(pred[1])


# **Confusion Matrix**

# In[ ]:


import pandas as pd
ytrue = pd.Series(np.argmax(y_test, axis= 1), name = 'ytrue')
ypred = pd.Series(np.argmax(pred, axis= 1), name = 'pred')
pd.crosstab(ytrue, ypred)


# ## Lernkurve plotten

# In[ ]:


class NeuralNetwork(object):
    def __init__(self, lr = 0.1):
        self.lr = lr

        self.w0 = np.random.randn(100, 784)
        self.w1 = np.random.randn(5, 100)


    def activation(self, x):
        return expit(x)

    def train(self, X, y):
        a0 = self.activation(self.w0 @ X.T)
        pred = self.activation(self.w1 @ a0)

        e1 = y.T - pred
        e0 = e1.T @ self.w1

        dw1 = e1 * pred * (1 - pred) @ a0.T / len(X)
        dw0 = e0.T * a0 * (1 - a0) @ X / len(X)

        assert dw1.shape == self.w1.shape
        assert dw0.shape == self.w0.shape

        self.w1 = self.w1 + self.lr * dw1
        self.w0 = self.w0 + self.lr * dw0

        # print("Kosten: " + str(self.cost(pred, y)))

    def predict(self, X):
        a0 = self.activation(self.w0 @ X.T)
        pred = self.activation(self.w1 @ a0)
        return pred

    def cost(self, pred, y):
        # SUM((y - pred)^2)
        s = (1 / 2) * (y.T - pred) ** 2
        return np.mean(np.sum(s, axis=0))

limits = [100, 1000, 3000, 9000, 10500]
test_accs = []
train_accs = []
for limit in limits:
    model = NeuralNetwork(0.25)

    for i in range(0, 100):
        for j in range(0, limit, 100):
           model.train(X_train[j:(j + 100), :] / 255., y_train_oh[j:(j + 100), :])


    y_test_pred = model.predict(X_test / 255.)
    y_test_pred = np.argmax(y_test_pred, axis=0)
    test_acc = np.mean(y_test_pred == y_test)

    y_train_pred = model.predict(X_train / 255.)
    y_train_pred = np.argmax(y_train_pred, axis=0)
    train_acc = np.mean(y_train_pred == y_train)

    test_accs.append(test_acc)
    train_accs.append(train_acc)



plt.plot(limits, train_accs, label="Training")
plt.plot(limits, test_accs, label="Test")

plt.legend()
plt.show()


# In[ ]:


test_acc


# In[ ]:


train_acc


# ## Lernrate plotten

# In[ ]:


class NeuralNetwork(object):
    def __init__(self, lr = 0.1):
        self.lr = lr

        self.w0 = np.random.randn(100, 784)
        self.w1 = np.random.randn(5, 100)


    def activation(self, x):
        return expit(x)

    def train(self, X, y):
        a0 = self.activation(self.w0 @ X.T)
        pred = self.activation(self.w1 @ a0)

        e1 = y.T - pred
        e0 = e1.T @ self.w1

        dw1 = e1 * pred * (1 - pred) @ a0.T / len(X)
        dw0 = e0.T * a0 * (1 - a0) @ X / len(X)

        assert dw1.shape == self.w1.shape
        assert dw0.shape == self.w0.shape

        self.w1 = self.w1 + self.lr * dw1
        self.w0 = self.w0 + self.lr * dw0

        # print("Kosten: " + str(self.cost(pred, y)))

    def predict(self, X):
        a0 = self.activation(self.w0 @ X.T)
        pred = self.activation(self.w1 @ a0)
        return pred

    def cost(self, pred, y):
        # SUM((y - pred)^2)
        s = (1 / 2) * (y.T - pred) ** 2
        return np.mean(np.sum(s, axis=0))


model = NeuralNetwork()

epochs = []
costs = []
accs = []

for i in range(0, 50):
    for j in range(0, 10500, 100):
        model.train(X_train[j:(j + 100), :] / 255., y_train_oh[j:(j + 100), :])

    cost = model.cost(model.predict(X_train), y_train_oh)

    y_test_pred = model.predict(X_test / 255.)
    y_test_pred = np.argmax(y_test_pred, axis=0)
    acc = np.mean(y_test_pred == y_test)

    epochs.append(i + 1)
    costs.append(cost)
    accs.append(acc)


import matplotlib.pyplot as plt


plt.plot(epochs, costs, label="Kosten")
plt.plot(epochs, accs, label="Genauigkeit")
plt.legend()
plt.show()


# In[ ]:


test_acc = np.mean(y_test_pred == y_test)


# ## Netzwerkgröße

# In[ ]:


class NeuralNetwork(object):
    def __init__(self, lr = 0.1, hidden_size = 100):
        self.lr = lr

        self.w0 = np.random.randn(hidden_size, 784)
        self.w1 = np.random.randn(5, hidden_size)


    def activation(self, x):
        return expit(x)

    def train(self, X, y):
        a0 = self.activation(self.w0 @ X.T)
        pred = self.activation(self.w1 @ a0)

        e1 = y.T - pred
        e0 = e1.T @ self.w1

        dw1 = e1 * pred * (1 - pred) @ a0.T / len(X)
        dw0 = e0.T * a0 * (1 - a0) @ X / len(X)

        assert dw1.shape == self.w1.shape
        assert dw0.shape == self.w0.shape

        self.w1 = self.w1 + self.lr * dw1
        self.w0 = self.w0 + self.lr * dw0

        # print("Kosten: " + str(self.cost(pred, y)))

    def predict(self, X):
        a0 = self.activation(self.w0 @ X.T)
        pred = self.activation(self.w1 @ a0)
        return pred

    def cost(self, pred, y):
        # SUM((y - pred)^2)
        s = (1 / 2) * (y.T - pred) ** 2
        return np.mean(np.sum(s, axis=0))

for hidden_size in [500, 600, 700, 800]:

    model = NeuralNetwork(0.3, hidden_size)

    for i in range(0, 25):
        for j in range(0, 10500, 100):
            model.train(X_train[j:(j + 100), :] / 255., y_train_oh[j:(j + 100), :])

        # cost = model.cost(model.predict(X_train), y_train_oh)

    y_test_pred = model.predict(X_test / 255.)
    y_test_pred = np.argmax(y_test_pred, axis=0)
    acc = np.mean(y_test_pred == y_test)

    print(str(hidden_size) + ": " + str(acc))


# In[ ]:


count=0
for i in range(0, len(X_test)):
    if y_test_pred[i] == 2 and y_test[i] ==1:
        count += 1
        plt.imshow(X_test[i].reshape(28, 28))
        plt.show()
        print(count)

