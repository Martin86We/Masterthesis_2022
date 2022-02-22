#!/usr/bin/env python
# coding: utf-8

# # Neuronales Netz (Funktionsweise)

# In diesem Abschnitt soll ein Einblick in den Aufbau und den Lernvorgang eines Neuronalen Netzes geschaffen werden.

# ## Verdeckte Schicht

# Bisher hatten wir nur ein Neuron. Da ein neuronales Netz aus mehreren solcher Neuronen aufgebaut ist, wollen wir uns in diesem Abschnitt damit befassen wie die einzelnen Neuronen zu einem Netz zusammen geschalten werden, wie diese einzelnen Neuronen arbeiten und wie es das Neuronale Netz schafft etwas zu lernen.

# **Ein Netz aus mehreren Neuronen:**
# Wir beginnen wieder mit einem einfachen Beispiel und verbinden ein paar Neuronen zu einem einfach NN:
# 
# - **Input Layer:** X1, X2, X3, b
# - **Hidden Layer** Neuron 1, Neuron 2, Neuron 3
# - **Output Layer** Neuron 4

# **Beispiel:**
# - X1: Anzahl Zylinder
# - X2: Leistung kw
# - X3: Gewicht kg
# 
# 
# Die Neuronen verteilen sich selbst auf verschiedene Features auf. 
# Jedes Neuron spezialisiert sich auf eine bestimmte Eigenschaft z.B.:
# 
# - Neuron 1: Kleinwagen oder SUV (relevant: X1, X2, X3)
# - Neuron 2: Preis
# - Neuron 3: Beschleunigung (relevant: X2,X3)
# 
# Relevante Verbindungen werden vom Algorithmus verstärkt und nicht benötigte Verbindungen werden ignoriert (Gewicht wird sehr klein oder Null).
# 
# Der Output-Layer soll z.B. den Verbrauch vorhersagen und wird entsprechend jene Verbindungen verstärken, die besonders großen Einfluss auf den Verbrauch haben.
# 
# Die Aktualisierung der Gewichte hat einen großen Einfluss auf diesen Vorgang.

# :::{figure-md} two layer net
# <img src="hiddenLayer_1.png" alt="nn" class="bg-primary mb-1" width="1000px">
# 
# Two-Layer-Neural Net.
# :::

# Weitere Informationen:
# [Neural network architecture](https://otexts.com/fpp2/nnetar.html).
# 

# :::{note} Die Neuronen im Hidden Layer übernehmen jeweils verschiedene Hilfsaufgaben.. Der Output Layer kombiniert der Ergebnisse aus dem Hidden Layer und gibt eine Vorhersage aus.
# :::

# :::{note}**Es gilt grundsätzlich:**
# - Neuronale Netze mit einem beliebig großen Hidden-Layer können jede beliebige Funktion approximieren.
# - je mehr Knoten das Netz besitzt, desto genauer kann es die math. Funktionen annähern.
# :::

# :::{note}Die Fähigkeit, jede beliebige math. Funktion anzunähern, macht neuronale Netze so mächtig. :::

# ## Gewichte aktualisieren

# Nach der Initialisierung der Gewichte. Wird ein Ausgangswert berechnet. Passt dieser "Vorhersagewert" nicht zum richtigen Ergebnis, dann müssen die Gewichte dementsprechend angepasst werden, so dass das Ergebnis stimmt.
# 
# In der Abb. werden w1 und w2 so lange erhöht bis der Vorhersagewert zum richtigen Wert passt.
# 
# Das Neuron gibt 0.5 aus, obwohl der richtige Wert 0.75 ist. Das bedeutet, das Modell ist noch nicht so gut an die Daten angepasst. Um das Modell den Daten besser anzupassen, stehen nur die Gewichte als Stellschrauben zur Verfügung und diese können nun Stückweise erhöht werden bis das Modell die Daten ausreichend approximiert hat siehe Abbildung unten.
# 

# :::{figure-md} two layer net
# <img src="weights_increase.png" alt="nn" class="bg-primary mb-1" width="600px">
# 
# 
# :::

# ## Kostenfunktion

# Die Aktualisierung der Gewichte wird mit Hilfe einer Kostenfunktion erreicht. Es gibt verschiedene Kostenfunktionen, eine davon ist die **"quadratische Fehlerfunktion"**.
# 
# Weitere Kostenfunktionen und Informationen [hier](https://www.analyticsvidhya.com/blog/2021/02/cost-function-is-no-rocket-science/).
# 

# :::{figure-md} two layer net
# <img src="weights_1.png" alt="nn" class="bg-primary mb-1" width="750px">
# 
# Kostenfunktion
# :::

# Die Quadrierung des Fehlers in der Kostenfunktion, bewirkt eine viel größere Bestrafung für größere Fehler.

# Werden nun wie üblich mehrere Datensätze trainiert, müssen die Gewichte nach jedem Datensatz angepasst werden. Je nachdem wie die Vorhersage vom Ergebnis y abweicht.

# :::{figure-md} two layer net
# <img src="weights_cost.png" alt="nn" class="bg-primary mb-1" width="1000px">
# 
# 
# :::

# Die Abbildung dient nur zur Veranschaulichung. Im Abschnitt Gradientenabstieg wird gezeigt wie die Kostenfunktion in Python minimiert wird.

# Eine Kostenfunktion dient zum Minimieren des Fehlers. Man stellt eine Funktion auf, die den Fehler zwischen Schätzung und richtigem Wert berechnet und sucht dann die zugehörigen Gewichte, die den Fehler minimal werden lassen. Die Kosten C werden als Funktion der Gewichte formuliert. Da man es bei NN’s meistens mit komplexeren Funktionen und sehr vielen Gewichten zu tun hat, kann man das nicht mehr analytisch lösen. Für so einen Fall eignet sich das Gradientenabstiegsverfahren.

# ## Gradientenabstieg

# Wichtig zum Verständnis des Trainings von neuronalen Netzen.
# 
# Wie aktualisiert der Computer mehrere tausend Gewichte?
# 
# Mit dem Gradientenabstiegsverfahren wird Schritt für Schritt das Minimum einer Funktion gesucht. Bei einfachen Funktionen kann man das noch analytisch lösen aber bei komplexeren Funtionen benötigt man das Gradientenabstiegsverfahren.

# Wie findet man das Minimum bei komplexen Funktionen wie im Bild rechts?
# Die Antwort lautet Gradientenabstiegsverfahren. Um dieses Verfahren näher zu erläutern beginnen wir wieder mit einem einfachen Fall,  
# Für den gradient descent gab es bzgl Lernrate usw eine gute Geogebra erklärung in einem anderen Kurs von Jannis, diese Erklärung hier einfügen.  

# :::{figure-md} gradient_descent
# <img src="gradient_descent.png" alt="nn" class="bg-primary mb-1" width="500px">
# 
# Die Gewichte der ersten Batch werden trainiert.
# :::

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 2 - 4 * x + 5


def f_ableitung(x):
    return 2 * x - 4


x = 5

#Schrittweite bzw Lernrate (lr):
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


# ::::{important}
# :::{note} Man kann nun mit der Lernrate experimentieren und z.B. einen großen Wert wählen. Es kann passieren, dass bei einer zu großen Schrittweite das Minimum übersprungen wird und somit nicht gefunden werden kann.
# :::
# ::::

# ```{admonition} Gut zu wissen
# :class: tip
# In hochdimensionalen Räumen spielen lokale Minimas keine Rolle mehr. Erklärung folgt.
# ```

# In der Praxis hat man es eher mit komplexeren Funktionen mit mehreren Minimas zu tun. Die Gefahr, in einem lokalen Minimum stecken zu bleiben, ist in höher Dimensionalen Räumen zu vernachlässigen. Erklärung kommt später noch.

# Geogebra dateien zeigen

# ## Stochastic Gradient Descent

# **Lernziele:**
# - Was ist eine Batch?
# - Wozu braucht man Batches?
# - Welche Größe sollten die Batches haben?

# Die Kosten für die gesamten Trainingsdaten zu berechnen und anschließend die Gewichte zu aktualisieren, würde bei sehr vielen Gewichten einen sehr hohen Rechenaufwand bedeuten, da die Kostenfunktion dann sehr viele variable Gewichte enthält. Deswegen geht man bei NN’s so vor, dass man nicht die Kosten für die gesamten Daten sondern nur für einzelne Batches berechnet und somit die Kosten approximiert. Das macht man dann für alle Batches und aktualisiert nach jedem Batch die Gewichte. So werden die Gewichte pro kompletten Durchgang mehrmals aktualisiert und nicht nur einmal am Ende eines kompletten Durchgangs. Das bringt allerdings mit sich, dass die Gewichte hin und her springen, im „ZickZack zum Minimum laufen“ Das führt insgesamt zu einem schnelleren Lernvorgang.

# :::{figure-md} two layer net
# <img src="gradient_batch1.png" alt="nn" class="bg-primary mb-1" width="1000px">
# 
# erste Batch
# :::

# Anstatt alle Gewichte mit einmal zu bestimmen ist es vorteilhafter den Trainingssatz in einzelne Batches aufzuteilen, somit wird die Berechnung schneller und die Gewichte werden nach jedem Durchlauf angepasst.

# Ablauf der Gewichtsanpassung für eine Batch:
# - Vorhersage machen
# - Kosten berechnen
# - Gewichte anpassen
# - dann nächste Batch

# :::{figure-md} two layer net
# <img src="gradient_batch2.png" alt="nn" class="bg-primary mb-1" width="1000px">
# 
# Zweite Batch
# :::

# **Begriffsdefinition:**
# - Batch: Eine Gruppe von Trainingsdaten innerhalb des Datensatzes
# - Epoche: Alle Batches wurden einmal durchlaufen
# - Lernrate: Schrittgröße

# ## Backpropagation

# NN’s wurden erst durch Backpropagation Leistungsstark. Dadurch erst war es möglich, das gesamte NN zu trainieren.
# Hier noch etwas Erklärung rund um Backpropagation und NN‘s allg. einfügen.
# Wie werden die Gewichte der vorherigen Schicht aktualisiert?
# Durch Backpropagation!

# - verleiht den NN's ihre Leistungsfähigkeit
# - verhalf zum Durchbruch von NN's
# - Idee aus den 70ern
# - vorher konnte man nur Teile eines Netzes trainieren

# Problematik beim Lernen von mehrschichtigen NN's:
# - Wie werden die Gewichte einer vorherigen Schicht aktualisiert?
# 
# Lösung: Backpropagation...
# 
# Die Gewichte des gesamten NN werden immer wieder aktualisiert, solange bis der Vorhersagewert so nah wie möglich am gewünschten Wert liegt. Wie das genau gemacht wird, soll in diesem Abschnitt gezeigt werden.

# **Ein grobes, einfaches Beispiel zur Backward-Propagation**:
# 
# Mit einem einfachen, groben Beispiel soll der Einstieg in das Verständis der Backpropagation erleichtert werden. Die Mathematik hinter der "Backpropagation" ist für Nicht-Informatiker/-mathematiker teilweise nicht so einfach zu verstehen. Für den Einstieg wird daher auf ein grobes Rechenbeispiel zurückgegriffen. Es soll an dieser Stelle zunächst nur der grobe Vorgang der BP veranschaulicht werden.
# 
# - Es wird eine Vorhersage mit dem NN gemacht, diese Vorhersage $\hat{y}$, weicht vom gewünschten / wahren Wert y ab
# - Die Abweichung e (Error) ist ein Maß dafür, wie stark die Vorhersage vom wahren Wert abweicht
# - Um die Abweichung zu minimieren müssen nun die Gewichte aktualisiert werden

# Der Fehler e wird nun an die Ausgänge des Hidden-Layer transformiert:

# :::{figure-md} backprop1
# <img src="backprop1.png" alt="nn" class="bg-primary mb-1" width="1000px">
# 
# Backpropagation
# :::

# **Initialisierung der Gewichte**

# Die Initialisierung kann darüber entscheiden, ob das NN trainieren kann oder nicht.
# Wie initialisieren wir die Gewichte? Mit Null wie im rechten Bild? Die Neuronen würden nur Nullen ausgeben. Das macht also keinen Sinn. Doch welche Werte soll man da am besten wählen? 
# Weiterhin dürfen die Gewichte nicht alle mit den gleichen Werten initialisiert werden. Das ist auch aktiver Forschungsgegenstand, denn bei mehrschichtigen Netzen wird es umso wichtiger die Gewichte „richtig“ bzw. nicht komplett falsch zu wählen. 
# Besser ist es, den Gewichten unterschiedliche Werte zu geben. Das können zufällige, eher kleine Werte sein. So ist sichergestellt, dass jedes Neuron eine andere Funktion berechnet. Somit kann dann auch die Backpropagation richtig arbeiten und die Gewichte anpassen.
# 

# :::{figure-md} backprop1
# <img src="backprop2.png" alt="nn" class="bg-primary mb-1" width="1000px">
# 
# Backpropagation2
# :::

# :::{figure-md} backprop1
# <img src="backprop3.png" alt="nn" class="bg-primary mb-1" width="1000px">
# 
# Backpropagation3
# :::

# In[ ]:




