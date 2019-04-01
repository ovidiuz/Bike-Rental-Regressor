#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 18:54:12 2019

@author: ovidiu
"""

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from keras.layers import Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf

# Metrica RMLSE
def RMSLE(Y_test, Y_predict):
    return tf.math.sqrt(tf.keras.backend.mean(tf.math.square(tf.math.log1p(Y_test) - tf.math.log1p(Y_predict))))

dataset = pd.read_csv('train.csv')
Y = dataset['count']

# Partea de preprocesare
# Coloana datetime este impartita in 3 coloane diferite: year, month, hour
# Am comentat aceasta parte deoarece executia ei dureaza foarte mult si
# am salvat datele rezultate in train2.csv

#X = dataset.drop('count', axis = 1)
#
#new = X["datetime"].str.split(" ", n = 1, expand = True) 
#X["date"] = new[0]
#X["hour"] = new[1]
#X.drop(columns=["datetime"], inplace = True)
#
#length = X["date"].size
#X["year"] = np.NaN
#X["month"] = np.NaN
#X["day"] = np.NaN
#
#for i in range(length):
#    date = datetime.strptime(X["date"][i], '%Y-%m-%d')
#    X["year"][i] = date.year
#    X["month"][i] = date.month
#    X["day"][i] = date.day
#    hour = datetime.strptime(X["hour"][i], "%H:%M:%S")
#    X["hour"][i] = hour.hour
#X.to_csv("train2.csv")

X = pd.read_csv('train2.csv')
X = X.drop('Unnamed: 0', axis = 1)
X = X.drop('date', axis = 1)
X = X.drop('casual', axis = 1)
X = X.drop('registered', axis = 1)

# Feature Selection

corr_matrix = pd.DataFrame(X).corr()
sns.heatmap(corr_matrix); 

X = X.drop('season', axis = 1)
X = X.drop('holiday', axis = 1)
X = X.drop('humidity', axis = 1)
X = X.drop('temp', axis = 1)

# OneHotEncoding

X = pd.get_dummies(X, columns=["weather", "month", "year", "hour"], drop_first=True)

# Scalarea datelor

sc_X = MinMaxScaler();
X = sc_X.fit_transform(X);

X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

scorer = make_scorer(RMSLE, greater_is_better = False);

# Reteaua propriu-zisa

model = Sequential()
model.add(Dense(42, input_dim=42, kernel_initializer='normal', activation='relu'))
model.add(Dense(21, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='linear'))
model.summary()

model.compile(loss=RMSLE, optimizer='adam', metrics=[RMSLE])

history = model.fit(X_train, Y_train, epochs=10, batch_size=1, verbose=1, validation_data=(X_test, Y_test))

print(history.history.keys())
plt.figure();
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

final_score=model.evaluate(x=X_test, y=Y_test, batch_size=1)
print(final_score)

