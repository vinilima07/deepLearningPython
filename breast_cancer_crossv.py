# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:11:27 2019

@author: Vinicius
"""


import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score


previsores = pd.read_csv('entradas_breast.csv')

classe = pd.read_csv('saidas_breast.csv')

def criarRede():
    
    classificador = Sequential()
    
    #camada de entrada
    classificador.add(Dense(units = 16, activation = 'relu', 
                            kernel_initializer = 'random_uniform', 
                            input_dim = 30))
    #dropout
    classificador.add(Dropout(0.2))
    
    #camada oculta
    classificador.add(Dense(units = 16, activation = 'relu', 
                            kernel_initializer = 'random_uniform'))
    
    #dropout
    classificador.add(Dropout(0.2))
    
    #camada de saida
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    
    #configuracao do otimizador
    otimizador = keras.optimizers.Adam(lr = 0.001, 
                                       decay = 0.0001, 
                                       clipvalue = 0.5)
    
    classificador.compile(otimizador, loss = 'binary_crossentropy',
                          metrics = ['binary_accuracy'])
    return classificador
    
classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 100,
                                batch_size = 10)

resultados = cross_val_score(estimator = classificador, 
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()





