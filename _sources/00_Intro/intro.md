![Img Alt 
Text](https://www.ionos.de/digitalguide/fileadmin/DigitalGuide/Teaser/ki-t.jpg)
[Quelle](https://www.ionos.de/digitalguide/fileadmin/DigitalGuide/Teaser/ki-t.jpg)


# Aufgabenstellung
**Anwendung neuronaler Netze in einem Sortierroboter**

Künstliche Intelligenz (KI) findet immer mehr Einzug in Maschinen. Ein Bereich der KI ist das Deep
Learning mit neuronalen Netzen, das sich z.$~$B. hervorragend zur Mustererkennung eignet. In dieser
Arbeit soll ein erster Schritt zu einem Sortierroboter gemacht werden, der im Endausbau geeignet ist,
Schrauben nach Art (Holz-, Sechskant-, Inbusschrauben etc.), Durchmesser (M5, M6, M8 etc.) und
Länge zu sortieren. In dieser Arbeit geht es aber zunächst nur um die Erkennung verschiedener
Bauarten. Dabei sollen vorerst Schrauben in gleicher Ansicht und aus einheitlicher Richtung
betrachtet werden.

In dieser Arbeit wird die Bilderkennung behandelt. Dazu sind folgende Schritte durchzuführen:
1. Erzeugung einer Bilddatenbank mit Bildern von Schrauben verschiedener Bauart. Die Bilder
sind mittels CAD zu erzeugen, selbst zu fotografieren oder aus dem Internet unter Beachtung
des Copyrights zu sammeln, falls dies erforderlich ist. Eine Mischung der drei Quellen ist ebenfalls möglich.
Die Bilddatenbank soll leicht um weitere Kategorien und um weitere Bilder erweitert werden
können.
2. Es ist, mit Python ein neuronales Netz zu erzeugen und für die Erkennung der
Schraubenbauarten zu optimieren. Dabei ist vor allem folgende Netzstruktur empirisch zu prüfen:
- 2 bis 3 Convolutional Layer
- 1 Pooling Layer
- 1 bis 2 fully connected Layer
3. Das entwickelte Programm ist klar zu strukturieren und vollständig zu kommentieren, so dass
Studierende, die später an dem Projekt weiterarbeiten, sich sehr gut in die Software
einarbeiten können.
4. Das Anlernen von neuronalen Netzen erfordert eine Menge Rechenzeit. Deshalb ist es
sinnvoll von Beginn an zu prüfen, ob durch die Nutzung einer GPU die Rechenzeit verringert werden kann.
5. Der Schwierigkeitsgrad für die Erkennungssoftware ist schrittweise zu steigern. Beginnend
mit Schrauben in identischer Lage, mit identischer Beleuchtung, hin zu Schrauben in
unterschiedlicher Lage und ggf. unterschiedlicher Beleuchtung.
Während des gesamten Projektes ist darauf zu achten, dass diese Arbeit der erste Schritt und damit
die Grundlage sein soll, auf der vermutlich im Laufe der nächsten Jahre viele weitere Projektarbeiten
aufbauen werden. Unter den zukünftigen Bearbeitern werden Studierende mit
wenig Programmiererfahrung sein.




















































