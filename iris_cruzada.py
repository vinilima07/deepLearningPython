# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:30:44 2019

@author: Vinicius
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()

#realizar o encod de uma string para integer
classe = labelencoder.fit_transform(classe)

#Converts a class vector (integers) to binary class matrix
classe_dummy = np_utils.to_categorical(classe)

#iris setosa     100
#iris virginica  010
#iris versicolor 001

def criar_rede():
    classificador = Sequential()
    classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
    classificador.add(Dense(units = 4, activation = 'relu'))
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                          metrics = ['categorical_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn = criar_rede,
                                epochs = 1000,
                                batch_size = 10)
resultados = cross_val_score(estimator = classificador, 
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()







