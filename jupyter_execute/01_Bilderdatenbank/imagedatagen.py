#!/usr/bin/env python
# coding: utf-8

# # Image Data Generator

# ## Preprocessing
# 

# Mit dem Image Data Generator werden wir unsere Bilder weiterverarbeiten. Wir werden den Datensatz künstlich erweitern um so das Modell robuster zu machen.

# ## Bilder in Numpy Array speichern

# Um sehr viele Bilder abzuspeichern in einer Form die für das CNN passend ist, werden die Bilder in ein Numpy Array gespeichert. Dieses Numpy Array ist zunächst 3 Dimensional, wird später aber noch auf 4D erweitert.
# 
# Wie man nun die Bilder oder besser gesagt die Pixelwerte in ein Numpy Array speichert zeigt der folgende Programmcode:

# ```
# # lob importieren
# import glob
# 
# # import numpy
# import numpy as np
# 
# # import Image from PIL
# from PIL import Image
# 
# # Dateinamen der Bilder in filelist speichern
# filelist = glob.glob('../pozi/*.jpg')
# 
# # Alle Bilder nacheinander öffnen und hintereinander in ein Numpy Array speichern
# x = np.array([np.array(Image.open(fname)) for fname in filelist])
# 
# ```

# ## Labels erstellen

# Wie erstellt man die Labels?

# ## Datensatz in Numpy Array exportieren

# (X_Train, y_Train, X_Test, y_Test) in ein Numpy Array gemeinsam speichern:

# In[ ]:




