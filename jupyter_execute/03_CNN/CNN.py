#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network

# **Why Convolutional Neural Networks**
# The main structural feature of RegularNets is that all the neurons are connected to each other. For example, when we have images with 28 by 28 pixels in greyscale, we will end up having 784 (28 x 28 x 1) neurons in a layer that seems manageable. However, most images have way more pixels and they are not grey-scaled. Therefore, assuming that we have a set of color images in 4K Ultra HD, we will have 26,542,080 (4096 x 2160 x 3) different neurons connected to each other in the first layer which is not really manageable. Therefore, we can say that RegularNets are not scalable for image classification. However, especially when it comes to images, there seems to be little correlation or relation between two individual pixels unless they are close to each other. This leads to the idea of Convolutional Layers and Pooling Layers.
# 
# [link](https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d)

# Ein Theorem aus dem Jahr 1988, das "Universal Approximation Theorem", sagt, dass jede beliebige, glatte Funktion, durch ein NN mit nur einem Hidden Layer approximiert werden kann. Nach diesem Theorem nach, würde dieses einfache NN bereits in der Lage sein jedes beliebige Bild bzw. die Funktion der Pixelwerte zu erlernen. Die Fehler und die lange Rechenzeit zeigen die Probleme in der Praxis. Denn um dieses Theorem zu erfüllen sind für sehr einfache Netze unendlich viel Rechenleistung, Zeit und Trainingsbeispiele nötig. Diese stehen i.d.R. nicht zur Verfügung. Für die Bilderkennung haben sich CNN's als sehr wirksam erwiesen. Die Arbeitsweise soll in diesem Abschnitt erläutert werden.
# Der Grundgedanke bei der Nutzung der Convolutional Layer ist, dem NN zusätzliches "Spezialwissen" über die Daten zu geben. Das NN ist durch den zusätzlichen Convolutional Layer in der Lage, spezielle Bildelemente und Strukturen besser zu erkennen. 
# 
# Es werden meist mehrere Convolutional Layer hintereinander geschalten. Das NN kann auf der ersten Ebene lernen, Kanten zu erkennen. Auf weiteren Ebenen lernt es dann weitere "Bild-Features" wie z.B. Übergänge, Rundungen o.ä. zu erkennen. Diese werden auf höheren Ebenen weiterverarbeitet.  
# 
# **Beispiel einer einfachen 1D-Faltung:**
# 
# Die beiden einfachen Beispiele sollen die Berechnung verdeutlichen. Die Filterfunktion wird auf die Pixel gelegt und Elementweise multipliziert. 
# Im folgenden Beispiel werden 3 Pixel eines Bildes verwendet. Die Ergebnisse sagen etwas über den Bildinhalt aus:
# 
# - positives Ergebnis: Übergang von hell zu dunkel   
# - negatives Ergebnis: Übergang von dunkel nach hell
# - neutrales Ergebnis: Übergang wechselnd, hell-dunkel-hell  oder dunkel-hell-dunkel 

# :::{figure-md} markdown-fig
# <img src="cnn_1d.png" alt="pozi" class="bg-primary mb-1" width="900px">
# 
# Eindimensionale Faltung
# :::

# Da ein Bild aus mehr als 3 Pixel besteht, muss die Filterfunktion über das gesamte Bild "geschoben" werden. Das folgende Beispiel demonstriert den Vorgang der Convolution im Fall eines eindimensionalen Filter. Der Filter besteht in diesem Fall wieder aus einem Zeilenvektor mit 3 Elementen. Der Filter wird nun Pixelweise über die Bildzeile geschoben, die Ergebnisse werden gespeichert und geben wiederum Aufschluss über die Bildstruktur.
# Die Ergebnisse zeigen wieder die enthaltene Bildstruktur: 
# 
# - 1: hell-dunkel
# - 0: hell-dunkel-hell
# - 0: dunkel-hell-dunkel
# - 1: hell-dunkel
# --1: dunkel-hell

# :::{figure-md} markdown-fig
# <img src="cnn_1d_long.png" alt="pozi" class="bg-primary mb-1" width="900px">
# 
# Eindimensionale Faltung mit mehreren Übergängen
# :::

# ## 2-Dimensionale Faltung

# In der Praxis werden in der Bilderkennung 2-dimensionale Filter verwendet, ein häufig verwendetes Format ist ein 3x3 Filter. Der Vorgang ist analog zum eindimensionalen Fall, der Filter wird über das gesamte Bild geschoben. Das folgende Beispiel zeigt einen Filter, der in der Lage ist, senkrechte Kanten zu erkennen.

# :::{figure-md} markdown-fig
# <img src="cnn_2d_a.png" alt="pozi" class="bg-primary mb-1" width="900px">
# 
# Eindimensionale Faltung mit mehreren Übergängen
# :::

# :::{figure-md} markdown-fig
# <img src="cnn_2d_b.png" alt="pozi" class="bg-primary mb-1" width="900px">
# 
# Eindimensionale Faltung mit mehreren Übergängen
# :::

# Die Werte der Filter bilden die Gewichte des Convolutional Layer. Diese Gewichte werden durch das Training selbst bestimmt und somit ist das CNN in der Lage, sich selbstständig auf relevante Features zu fokussieren. 

# **Im folgenden noch weitere Ergebnisse für bestimmte Bildstrukturen:**
# 

# :::{figure-md} markdown-fig
# <img src="cnn_2d_c.png" alt="pozi" class="bg-primary mb-1" width="900px">
# 
# Eindimensionale Faltung mit mehreren Übergängen
# :::

# :::{figure-md} markdown-fig
# <img src="cnn_2d_d.png" alt="pozi" class="bg-primary mb-1" width="900px">
# 
# Eindimensionale Faltung mit mehreren Übergängen
# :::

# In[1]:





# Mit Hilfe eines CNN-Layer bekommt das neuronale Netz ein "Verständnis" für Bilder "eingebaut". Das NN ist somit auf die Erkennung von Bildern spezialisiert und demensprechend Leistungsfähiger als ein NN ohne dieses Bildverständnis.
# 
# - Kantenerkennung
# 

# Das CNN besitzt gegenüber dem neuronalem Netz eine Intuition darüber was ein Bild ist.

# Das Neuronale Netz kann auf der ersten Ebene lernen, Kanten zu erkennen. Diese Ebene ist dann für die Kantenerkennung zuständig. Kante ist Kante egal wo auf dem Bild. Diese "Features" werden in den nachfolge Schichten weiterverarbeitet.

# ### Beispiel einer einfachen Convolution:

# https://medium.com/swlh/image-processing-with-python-convolutional-filters-and-kernels-b9884d91a8fd

# Die Filter oder Kernels gibt man nicht vor sondern lässt die Werte vom Convolutional Layer ermitteln. Die Kernels werden dabei so bestimmt dass sie für das Problem am meisten Sinn machen.

# Wir möchten nicht nur vertikale Kanten finden, sondern auch schräge und waagerechte. Da jeder Filter für ein bestimmtes Feature zuständig ist, benötigt das CNN mehrere solcher Filter um alle relevanten Zusammenhänge extrahieren zu können. Die Anzahl an Filter die wir bereitstellen hängt von den Daten ab und ist ein Hyperparameter den man tunen muss.

# ## CNN mit Keras

# Wir wollen nun ein CNN mit Keras entwickeln.

# In[1]:


# Vorstellung: MNIST-Daten!
# http://yann.lecun.com/exdb/mnist/
# FashionMNIST: https://github.com/zalandoresearch/fashion-mnist

import gzip
import numpy as np
import numpy as np
from numpy import load
from tensorflow.keras.utils import to_categorical


X_train = load('../02_NN/Dataset/X_train.npy').astype(np.float32)#.reshape(-1, 784)
y_train = load('../02_NN/Dataset/y_train.npy')


#oh = OneHotEncoder()
#y_train_oh = oh.fit_transform(y_train.reshape(-1, 1)).toarray()

X_test=load('../02_NN/Dataset/X_test.npy').astype(np.float32)#.reshape(-1, 784)
y_test=load('../02_NN/Dataset/y_test.npy')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[2]:


print(y_train)


# In[3]:


print(X_train.shape)


# Das Format der Daten passt noch nicht zum geforderten Eingangsformat.
# Das CNN verlangt

# Bei einem Wert am Ausgang zwischen 0 und 1 verwendet man "binary crossentropy". Hat man mehrere Werte / Kategorien am Ausgang, dann verwendet man categorical crossentropy.

# ## stochastic gradient descent

# In[4]:


# CNN!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
#model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(5, activation="softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    X_train.reshape(10500,28,28,1),
    y_train,
    epochs=20,
    batch_size=500)


# ## rmsprop

# In[5]:


# CNN!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
#model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(5, activation="softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    X_train.reshape(10500,28,28,1),
    y_train,
    epochs=20,
    batch_size=500)


# ## Two Conv2D Layer

# In[6]:


# CNN!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(5, activation="softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    X_train.reshape(10500,28,28,1),
    y_train,
    epochs=20,
    batch_size=500)


# In[ ]:




