#!/usr/bin/env python
# coding: utf-8

# # Einleitung

# Wer kennt es nicht, es ist gerade wieder Lockdown und man möchte ein Heimwerker Projekt umsetzen. Nun stellt man fest, dass man von einigen Schrauben nicht genügend vorrätig hat und Nachschub besorgen muss. Da alle Baumärkte geschlossen haben hat man nur noch die Möglichkeit online die gewünschten Exemplare zu ordern. Der Händler des Vertrauens benötigt lediglich die genaue Bezeichnung der benötigten Schrauben und da liegt das Problem, woher soll man die Bezeichnung der Schraube wissen ohne Seitenweise Kataloge zu studieren? Die Lösung wäre eine automatische Schraubenerkennung. Ein Foto von der Schraube als Eingabe und die Bezeichnung der Schraube als Output.  

# Eine Suche nach Objekterkennung bei Google ergab:
# - Object recognition is a computer vision technique for identifying objects in images or videos. Object recognition is a key output of deep learning and machine learning algorithms. When humans look at a photograph or watch a video, we can readily spot people, objects, scenes, and visual details.

# Wir wollen also Objekte, in unserem Fall Schrauben in Bildern erkennen und dafür machine learning und deep learning Techniken verwenden.

# ## Objekte in Bildern erkennen

# Ziel ist es, Schrauben auf einem Bild zu erkennen. Dabei stellen sich folgende Fragen:
# - Welche Eigenschaften hat eine Schraube und welche wollen / können wir erkennen?
# - Was gibt es für Schraubenarten und wodurch unterscheiden sie sich?
# - Wie werden die Bilder erzeugt und wie sollten diese beschaffen sein?

# Um mit einfachen Modellen starten zu können sollte auch die Lernaufgabe zunächst so einfach wie möglich gestaltet sein. Es wäre sicher nicht die richtige Vorgehensweise, direkt mit der höchsten Komplexität zu starten. So sind wir in der Lage den dahinter stehenden Mechanismus besser zu verstehen und nachvollziehen zu können.
# 
# Um schnell erste eigene Modelle zu entwickeln lassen wir uns vom MNIST Datensatz inspirieren. Dieser ist sehr bekannt und wurde schon sehr häufig mit den verschiedensten Modellen (SVM,CNN,Logistic Regression,..) bearbeitet. Der Vorteil besteht darin, dass schon sehr viele Informationen und fertige Modelle existieren.
# 
# Die Idee ist nun, fertige Modelle des MNIST Datensatzes zu nehmen und diese mit einem Eigenen Datensatz zu trainieren. So kann man schnell Erfahrungen sammeln und durch den Programmcode die Vorgänge nachvollziehen.
# 
# Um einfach zu beginnen konzentrieren wir uns zunächst auf eine Eigenschaft der Schraube, dem Kopf bzw. der Antriebsart (Kreuzschlitz, Torx, Sechskant, usw.) und nehmen dafür einfache Grafiken die sehr gut zu unterscheiden sind damit sich das Modell nur auf die Form konzentrieren kann und nicht von Lichteinflüssen wie in realen Umgebungen beeinflusst wird. Die Auswahl viel auf folgende gängigen Antriebsarten:
# - Innensechskant (Inbus)
# - Philips (Kreuzschlitz)
# - Pozidriv (Kreuzschlitz)
# - Sechskant
# - Torx
# 
# Diese sind, zumindest für das menschliche Auge, gut zu unterscheiden und sollten von einem Modell welches mit den MNIST-Daten funktioniert auch gut erkannt werden können.
# 
# Es gibt jedoch auch schon gewisse Schwierigkeiten:
# - Innensechskant und Torx ähneln sich,schwarzer Ring auf weißem Hintergrund, Innenkontur auch ähnlich
# - Philips-Kreuz und Pozidriv-Kreuz sind nur durch die Einkerbungen in der Pozidriv zu unterscheiden
# - Das Sechskant-Profil ähnelt keinem Profil und unterscheidet sich stark von allen anderen, schwarze Fläche mit sechseckiger Außenkontur
# 
# Es ist zu erwarten, dass die Modelle mit dem Sechskant weniger Schwierigkeiten haben werden als mit den anderen Profilen.
# 
# So sollte dieser Datensatz einfach genug für den Anfang sein aber bringt auch schon die ersten schwierigkeiten mit sich, welche die Performance der verschiedenen Modelle auf die Probe stellen.

# In[1]:


from PIL import Image

im1 = Image.open(r'path where the PNG is stored\file name.png')
im1.save(r'path where the JPG will be stored\new file name.jpg')

