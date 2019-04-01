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

def RMSLE(Y_test, Y_predict):
    return np.sqrt(np.mean(np.square(np.log1p(Y_test) - np.log1p(Y_predict))))

dataset = pd.read_csv('train.csv')
Y = dataset['count']

#X = dataset.drop('count', axis = 1)
##X = pd.read_csv('train2.csv')
##X = X.drop(columns=["Unnamed: 0"], axis=1)
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

X = pd.get_dummies(X, columns=["season", "weather", "year", "month", "hour"], drop_first=True)

sc_X = MinMaxScaler();
X = sc_X.fit_transform(X);

corr_matrix = pd.DataFrame(X).corr()
sns.heatmap(corr_matrix); 

X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

pca = PCA(n_components = 25, random_state=0)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

grid_params = [
        
            {
                'booster': ['gbtree'],
                'objective':['reg:linear'],
                'learning_rate': [.03, 0.05, .07],
                'max_depth': [5, 6, 7, 10, 30],
                'min_child_weight': [1, 2, 3, 4],
                'silent': [1],
                'subsample': [0.7],
                'colsample_bytree': [0.7],
                'n_estimators': [100, 500],
            }
            
            ]

scorer = make_scorer(RMSLE, greater_is_better = False);

grid_search = GridSearchCV(estimator=XGBRegressor(), param_grid=grid_params, scoring=scorer, cv=5, n_jobs=-1, refit=True) 
grid_search.fit(X_train, Y_train)


best_score = grid_search.best_score_
best_params = grid_search.best_params_
    


print('Parametrii optimi sunt :');
print(best_params);
print('Cel mai bun scor pentru GridSearch: ');
print(best_score);

classifier = grid_search.best_estimator_;
classifier.fit(X_train,Y_train);
Y_predict = classifier.predict(X_test)
final_score = RMSLE(Y_predict, Y_test)
print('Scorul final pe teste(RMLSE): ');
print(final_score);
