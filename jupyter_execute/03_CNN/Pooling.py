#!/usr/bin/env python
# coding: utf-8

# # Pooling Layer

# Ein Pooling Layer, hat die Aufgabe, das CNN "toleranter" gegen geringe Abweichungen zu machen. Im folgenden Beispiel wurde die Kante stückweise um insgesamt 2 Pixel nach rechts verschoben. Auf einem Bild sollte es für das Erkennen keine so große Rolle spielen aber für den Convolutional-Layer spielt es eine Rolle in den Ergebnissen. Es ist nicht tolerant gegenüber Verschiebungen.

# **Jede Verschiebung der senkrechten Kante, erzeugt eine andere Featur-Map:**

# :::{figure-md} markdown-fig
# <img src="pooling_1.png" alt="pozi" class="bg-primary mb-1" width="1000px">
# 
# Kantenerkennung ohne Pooling.
# :::

# :::{figure-md} markdown-fig
# <img src="pooling_2.png" alt="pozi" class="bg-primary mb-1" width="1000px">
# 
# Kantenerkennung mit Pooling.
# :::

# In[ ]:





# **Vorteile vom Pooling:**
# - bessere Generalisierung
# - geringere Anzahl an Ausgabeknoten durch Dimensionsreduktion
# - lernen geht schneller

# In[ ]:




