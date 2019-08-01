# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:11:27 2019

@author: Vinicius
"""


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas_breast.csv')

classe = pd.read_csv('saidas_breast.csv')

classificador = Sequential()
#camada de entrada
classificador.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal', 
                        input_dim = 30))
#dropout
classificador.add(Dropout(0.2))

#camada oculta
classificador.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal'))
#dropout
classificador.add(Dropout(0.2))

#camada de saida
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size = 10, epochs = 100)    

classificador_json = classificador.to_json()

with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)

classificador.save_weights('classificador_breast.h5')



