#!/usr/bin/env python
# coding: utf-8

# # Datensatz aus Video

# :::{figure-md} markdown-fig
# <img src="pozi_quer.PNG" alt="pozi" class="bg-primary mb-1" width="1000px">
# 
# This is a Pozidriv **Schraube**!
# :::

# Hier ist ein **[Link](https://towardsdatascience.com/how-do-you-know-you-have-enough-training-data-ad9b1fd679ee)**.

# Das Bild der Schraube stammt aus einem Video, welches mit einem Iphone 12 pro aufgenommen wurde. Die Aufnahme ist sehr detailreich, das Pozidriv Profil sowie auch Beschädigungen, Rost und Ablagerungen auf der Oberfläche, sind gut zu erkennen.
# 

# Nachdem wir die Videos aufgenommen haben und die Softwareumgebung installiert ist, beginnen wir mit dem Python Programm zum Extrahieren von Bildern aus Videos:

# :::{note}
# Es stellt sich die Frage wie detailreich die Schraubenbilder sein müssen um eine gute Erkennung zu ermöglichen?
# :::
# 

# ## Extracting and Saving Video Frames using OpenCV-Python

# In[1]:



# OpenCV importieren:

import cv2



# path = 'relativer Speicherpfad / Dateiname':

path = 'pozi/pozi'



# Video laden:

cap = cv2.VideoCapture('pozi.MOV')
i = 0


# Prüfen ob ein Video geladen wurde:

if cap.isOpened() == False:
    print('ERROR: Datei nicht gefunden')

    
    
# Die Frames des Videos lesen:    
    
while(cap.isOpened()):
    ret, frame = cap.read()
     
    # sobald keine Frames mehr gelesen werden können (ret==False) wird abgebrochen:
    if ret == False:
        break
     
    # Die Frames speichern
    cv2.imwrite(path+str(i)+'.jpg', frame)
    i += 1
 

cap.release()
cv.destroyAllWindows()


# ## Crop Image

# In[ ]:


from PIL import Image

img = Image.open('dataset_from_video/pozidriv0.jpg')
plt.imshow(img)


# In[ ]:





# ## Resize Image

# In[ ]:


import cv2

img = cv2.imread('pozi/pozi800.jpg', cv2.IMREAD_UNCHANGED)

print('Original Dimensions : ',img.shape)

scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

#resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

print('Resized Dimensions : ',resized.shape)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




