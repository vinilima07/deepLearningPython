# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:30:44 2019

@author: Vinicius
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix


labelencoder = LabelEncoder()

#realizar o encod de uma string para integer
classe = labelencoder.fit_transform(classe)

#Converts a class vector (integers) to binary class matrix
classe_dummy = np_utils.to_categorical(classe)

#iris setosa     100
#iris virginica  010
#iris versicolor 001

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size = 0.25)

classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                      metrics = ['categorical_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, 
                  batch_size = 10, epochs = 1000)

#[1] o valor final da 'loss', [2] a % de acerto na base de teste
resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.8)

previsoes_tmp = [np.argmax(t) for t in previsoes]
classe_teste_tmp = [np.argmax(t) for t in classe_teste]

matriz = confusion_matrix(previsoes_tmp, classe_teste_tmp)