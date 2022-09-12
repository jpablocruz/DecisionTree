from json import load
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree

iris = pd.read_csv("iris.csv")

#Modelo Bagging
#Eliminamos todos los valores que no nos sirven o ayudan con la finalidad de mejorar nuestro modelo
#En este caso eliminaremos el identificador
iris.drop(["Id"], axis = 1, inplace=True)

valoresX = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]] #caracteristicas del modelo o variables x
valoresY = iris[["Species"]] #Nuestra variable objetivo es la especie de flor

train_setX, test_setX, train_setY, test_setY = train_test_split(valoresX, valoresY,test_size=0.40, random_state=40)
trainXt, testXv, trainYt, testYv = train_test_split(valoresX, valoresY, test_size= 0.15, random_state=40)

iris_classifier = DecisionTreeClassifier(criterion="gini", min_samples_split=3, max_depth=9) #creamos el modelo de arbol de decision con la libreria de sckikitlearn
iris_classifier.fit(trainXt, trainYt) #agregamos las variables x y la variable objetivo a la funcion de fit, la cual se encarga de tomar los datos del training set como argumento
#Daremos uso de la metrica de cross_validation para saber si el modelo esta entrenado correctamente+
puntajes = cross_val_score(iris_classifier, trainXt, trainYt, cv=3, scoring="accuracy")
print(puntajes)
print(puntajes.mean())