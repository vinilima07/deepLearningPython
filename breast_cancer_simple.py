# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:11:27 2019

@author: Vinicius
"""


import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv('entradas_breast.csv')

classe = pd.read_csv('saidas_breast.csv')

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()

#camada de entrada
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', 
                        input_dim = 30))
#camada oculta
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform'))
#camada de saida
classificador.add(Dense(units = 1, activation = 'sigmoid'))

#configuracao do otimizador
otimizador = keras.optimizers.Adam(lr = 0.001, 
                                   decay = 0.0001, 
                                   clipvalue = 0.5)

classificador.compile('adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, 
                  batch_size = 10, epochs = 100)

peso0 = classificador.layers[0].get_weights()
print(peso0)

previsoes = classificador.predict(previsores_teste)

previsoes = (previsoes > 0.5)

precisao = accuracy_score(classe_teste, previsoes)

matrix = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)



