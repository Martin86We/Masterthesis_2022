#!/usr/bin/env python
# coding: utf-8

# # Datensatz 1 (Schraubenköpfe)

# ## Bilder anzeigen

# Der Datensatz 1 wird aus nur 5 Bildern erzeugt. Die Bilder stellen die Schraubenköpfe Symbolartig dar. Desweiteren sind die Bilder nur 28 x 28 Pixel groß.

# In[1]:


# importieren der Matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pozidriv = mpimg.imread('0_Schraubenkopfbilder_28x28/pozidriv/pozidriv 28x28_gray.jpg')
philips = mpimg.imread('0_Schraubenkopfbilder_28x28/philips/philips 28x28_gray.jpg')
innensechskant = mpimg.imread('0_Schraubenkopfbilder_28x28/innensechskant/innensechskant 28x28_gray.png')
sechskant = mpimg.imread('0_Schraubenkopfbilder_28x28/sechskant/sechskant 28x28_gray.jpg')
torx = mpimg.imread('0_Schraubenkopfbilder_28x28/torx/torx 28x28_gray.jpg')


# In[3]:


plt.figure(figsize=(30, 6))
plt.subplot(1, 5, 1)
plt.title('Pozidriv')
plt.imshow(pozidriv, cmap='gray')

plt.subplot(1, 5, 2)
plt.title('philips')
plt.imshow(philips, cmap='gray')

plt.subplot(1, 5, 3)
plt.title('innensechskant')
plt.imshow(innensechskant, cmap='gray')

plt.subplot(1, 5, 4)
plt.title('torx')
plt.imshow(torx, cmap='gray')

plt.subplot(1, 5, 5)
plt.title('sechskant')
plt.imshow(sechskant, cmap='gray')

plt.suptitle('Datensatz Klassen', fontsize=20)
#plt.subplots_adjust(left=0.2, wspace=0.4, top=0.8)
plt.show()


# Der Datensatz dient zum Trainieren und Testen der künstlichen neuronalen Netze. Es gibt verschiedene Wege einen Datensatz von Schraubenbildern zu erzeugen:
#  - virtuelle 3D Objekte mit Hilfe von CAD darstellen und daraus Bilder mit unterschiedlichen Ausrichtungen erzeugen
#  - Schraubenbilder aus Herstellerkatalogen oder dem Internet extrahieren
#  - Eigene Bilder mit einer Kamera anfertigen
# 
# In dieser Arbeit werden die letzteren beiden Varianten vorgestellt. Zunächst wird gezeigt, wie aus wenigen externen Bidern ein Datensatz künstlich erzeugt werden kann. Dieser künstlich erzeugte Daten besteht aus sehr wenigen Ursprungsbildern, die künstlich transformiert werden um somit weitere Bilder zu generieren. Die Transformation wird im Abschnitt "Datensatz aus wenigen Bildern" noch näher beschrieben.

# ## Datensatz aus wenigen Bildern

# Dieser Erste Datensatz dient dazu, zu zeigen wie vorhandene Bilder künstlich leichten Veränderungen unterzogen werden können um somit dem Netz einen umfangreicheren Trainingssatz bereit zu stellen. Desweiteren werden in diesem künstlichen Datensatz lediglich verschiedene Schraubenkopfformen mit gleicher Ausrichtung verwendet. Die Folge ist ein Datensatz, der einfach genug ist um Regressionsmodelle zu Einstieg zu verwenden.

# Als Einführung in das Thema Bilderkennung mit CNN's, wird eine einfache Bilderkennung mit Hilfe der Logistischen Regression erstellt. Die Log. Regr. ist sehr viel weniger leistungsfähig als ein CNN, aus diesem Grund wird ein einfacher Datensatz aus Symbol-Bildern von 5 Schraubenarten erstellt.

# Aus diesen Bildern wird nun ein Datensatz erzeugt. Die Bilder werden mit Hilfe des "Image Data Generators" transformiert, d.h durch Rotation, Zoom, Verschiebung werden neue Varianten der Bilder generiert.

# ## Image Data Generator Vorschau

# In[4]:


img = torx.reshape(1,28,28,1)


# In[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ImageDataGenerator konfigurieren
datagen = ImageDataGenerator(rotation_range=30,
                             fill_mode='constant',cval=255,
                             width_shift_range=2.0,
                             height_shift_range=2.0,
                             shear_range=0.0,
                             zoom_range=0.20,
                             channel_shift_range=0.0,
                             horizontal_flip=False,
                             vertical_flip=False,
                             validation_split=0.0,)


# von: https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/
# iterator
aug_iter = datagen.flow(img, batch_size=1)

# generate samples and plot
fig, ax = plt.subplots(nrows=1, ncols=8, figsize=(15,15))

# ncols
# generate batch of images
for i in range(8):

	# convert to unsigned integers
	image = next(aug_iter)[0].astype('uint8')
 
	# plot image
	ax[i].imshow(image, cmap = 'gray')
	ax[i].axis('on')


# ## Image Data Generator (Speichern)

# In[6]:


#img = pozidriv.reshape(1,28,28,1)


# In[7]:


itr = datagen.flow(
    
    img,
    y=None,
    batch_size=32,
    shuffle=True,
    sample_weight=None,
    seed=None,
    save_to_dir='dataset_28x28/torx',
    save_prefix='torx',
    save_format='png',
    subset=None
)

# Samples in Ordner schreiben
for i in range(0,20):
    itr.next()


# Die Bilder der Klassen können unterschiedlich stark transformiert werden.

# In[ ]:





# Die Bilder wurden in das Array **X** geschrieben.

# ## Bilder Datensatz mit Labels (manuell)

# Die künstlich erzeugten Bilder wurden nach Klassen sortiert.  
# Nun wird aus dieser Ordner Struktur ein Datensatz mit Labels erzeugt (image_dataset).
# Die folgende Ordnerstruktur liegt vor:
# 
# main_directory/  
# ...category_a/  
# ......a_image_1.jpg  
# ......a_image_2.jpg  
# ...category_b/  
# ......b_image_1.jpg  
# ......b_image_2.jpg  

# In[8]:


DATADIR = 'dataset_28x28'
CATEGORIES = ['philips', 'pozidriv','torx']


# In[9]:


#DATADIR = '0_Schraubenkopfbilder_28x28'
#CATEGORIES = ['innensechskant', 'philips', 'pozidriv', 'sechskant', 'torx']


# In[10]:


# training_data anlegen
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
image_dataset = []

# Bildgröße
IMG_SIZE=28

# Bilder und Kategorien in ein ... speichern:

def create_image_dataset():
    for category in CATEGORIES:  # jede Klasse

        path = os.path.join(DATADIR,category)  # create path to 
        class_num = CATEGORIES.index(category)  # get the classification ( 0= 1=

        for img in tqdm(os.listdir(path)):  # iterate over each image per Category
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                image_dataset.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path,img))
            except Exception as e:
                print("general exception", e, os.path.join(path,img))

create_image_dataset()

print(len(image_dataset))


# In[ ]:


X = []
y = []

for features,label in image_dataset:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# ## Datensatz mischen

# In[ ]:


from sklearn.utils import shuffle
import numpy as np


X, y = shuffle(X, y)


# ## Datensatz Vorschau

# In[139]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

nrow = 3
ncol = 20


fig = plt.figure(figsize=(ncol+1, nrow+1)) 

gs = gridspec.GridSpec(nrow, ncol,
         wspace=0.0, hspace=0.0, 
         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
         left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
im = 0
for i in range(nrow):
    for j in range(ncol):
        ax= plt.subplot(gs[i,j])
        ax.imshow(X[im,:,:,0],cmap='gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        im +=1

plt.show()


# In[ ]:


X.reshape(-1, IMG_SIZE, IMG_SIZE).shape


# ## Datensatz aufteilen

# Die Bilder **X** und die entsprechenden Labels **y** werden nun noch aufgeteilt.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ## Datensatz untersuchen

# Wie sieht der Datensatz aus? wurden die Bilder gemischt oder sind sie immer noch nach Klassen sortiert?

# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(len(y_train))
print(len(y_test))


# In[ ]:


i=65
print(y_test[i])
plt.imshow(X_test[i],cmap='gray',vmin=0,vmax=255)
plt.show


# ## Labels aus Dateinamen

# Nun liegen alle generierten Bilder in einem Ordner. Jetzt werden die Klassen über die Dateinamen abgefangen und in der gleichen Reihenfolge in der Liste "categories" abgelegt.

# In[ ]:


import os

data_path = "dataset_28x28"
for img_filename in os.listdir(data_path):
    if img_filename.startswith("_0"):
        null, category, rand_str, = img_filename.split('_')
        nullen.append(null)
        categories.append(category)
        rand_strs.append(rand_str)
        
for img_filename in os.listdir(data_path):
    if img_filename.startswith("_1"):
        null, category, rand_str, = img_filename.split('_')
        nullen.append(null)
        categories.append(category)
        rand_strs.append(rand_str)

for img_filename in os.listdir(data_path):
    if img_filename.startswith("_2"):
        null, category, rand_str, = img_filename.split('_')
        nullen.append(null)
        categories.append(category)
        rand_strs.append(rand_str)       
        

for img_filename in os.listdir(data_path):
    if img_filename.startswith("_3"):
        null, category, rand_str, = img_filename.split('_')
        nullen.append(null)
        categories.append(category)
        rand_strs.append(rand_str)
        
for img_filename in os.listdir(data_path):
    if img_filename.startswith("_4"):
        null, category, rand_str, = img_filename.split('_')
        nullen.append(null)
        categories.append(category)
        rand_strs.append(rand_str)   


# In[ ]:


print(img_filename)
print(categories)
len(categories)


# In[ ]:


y=categories


# ## Samples von einzelnem Bild

# In[ ]:


# example of horizontal shift image augmentation
# from: https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from matplotlib import pyplot
# load the image
img = load_img('0_Schraubenkopfbilder_28x28/pozidriv/pozidriv 28x28_gray.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)


# In[ ]:


# prepare iterator
it = gen.flow(samples, batch_size=1)

# generate samples and plot

for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
    
    
# show the figure
pyplot.show()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'gen.flow')


# In[ ]:





# In[ ]:


from numpy import save, load
# define data
#data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to npy file
save('Dataset_vid/X_train.npy', X_train)
save('Dataset_vid/y_train.npy', y_train)
save('Dataset_vid/X_test.npy', X_test)
save('Dataset_vid/y_test.npy', y_test)

