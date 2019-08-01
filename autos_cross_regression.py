# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:57:53 2019

@author: Vinicius
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.wrappers.scikit_learn import KerasRegressor

base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

#**Tratamento dos dados** #

#conta os dados para verificar sua relevancia
base['name'].value_counts()

#deleta uma coluna
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)
base = base.drop('name', axis = 1)
base = base.drop('seller', axis = 1)
base = base.drop('offerType', axis = 1)

i1 = base.loc[base.price <= 10]

base.price.mean()
base = base[base.price > 10]
base = base[base.price > 10]
base = base[base. price < 35000]


base.loc[pd.isnull(base['vehicleType'])] #limousine

base.loc[pd.isnull(base['gearbox'])] # manuell

base.loc[pd.isnull(base['model'])] # golf

base.loc[pd.isnull(base['fuelType'])] #benzin

base.loc[pd.isnull(base['notRepairedDamage'])] #nein

valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf',  'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}

base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values

preco_real = base.iloc[:, 0].values

#transformacao de atributos categoricos em numericos
labelEncoder_previsores = LabelEncoder()
previsores[:, 0] = labelEncoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelEncoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelEncoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelEncoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelEncoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelEncoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelEncoder_previsores.fit_transform(previsores[:, 10])

#cricao de variaveis do tipo dummy
oneHotEncoder = OneHotEncoder(categorical_features = [0, 1, 3, 5, 8, 9, 10])
previsores = oneHotEncoder.fit_transform(previsores).toarray()

def criar_rede():
    regressor = Sequential()
    regressor.add(Dense(units = 158, activation = 'relu'))
    regressor.add(Dense(units = 1, activation = 'linear'))
    regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam',
                      metrics =['mean_absolute_error'])
    regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)
    
    return regressor

regressor = KerasRegressor(build_fn = criar_rede,
                           epochs = 100, 
                           batch-size = 300)
resultados = cross_val_score(estimator = regressor, 
                             X = previsores, y = preco_real,
                             cv = 10, scoring = 'mean_absolute_error')
media = resultados.mean()
desvio = resultados.std()

    





