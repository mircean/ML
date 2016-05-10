import sys
from datetime import datetime
from math import sqrt
import threading
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.core import DMatrix
from xgboost.training import train, cv
from xgboost.plotting import plot_importance

from cv_common import my_cv, my_predict, func_cv_1, func_predict_1

np.random.seed(0)

#Loading data
print('load data', datetime.now())
df_train = pd.read_csv('Dataset\\train.csv')

labels = df_train['TARGET'].values
df_train = df_train.drop(['TARGET'], axis=1)
piv_train = df_train.shape[0]

df_all = pd.read_csv('Dataset\\df_all.csv')

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
y = labels
X_predict = vals[piv_train:]

models = []
models.append([0.02, 5, 0.6815, 0.701, 0, 6, 1, 0])
models.append([0.01, 5, 0.6815, 0.701, 0, 6, 1, 0])
models.append([0.02, 5, 0.6815, 0.701, 0, 6, 1, 0.1])
models.append([0.02, 5, 0.6815, 0.701, 0, 7, 1, 0])     #0.84261423           
'''
models.append([0.02, 5, 0.6815, 0.701, 0, 6, 1, 0.3])   #0.84260816      
models.append([0.02, 5, 0.6815, 0.701, 0, 6, 2, 0])     #0.84258153      
models.append([0.02, 5, 0.6815, 0.701, 0, 8, 1, 0])     #0.84257286      
models.append([0.0202064, 5, 0.6815, 0.701, 0, 6, 1, 0])#0.84257033      
models.append([0.02, 5, 0.7, 0.7, 0, 6, 1, 0])          #0.84256522      
models.append([0.02, 5, 0.6815, 0.701, 0, 6, 1, 1])     #0.84255844      
models.append([0.02, 5, 0.6815, 0.701, 0, 5, 1, 0])     #0.84251736      
models.append([0.02, 5, 0.6815, 0.701, 0, 10, 1, 0])    #0.84249619      
models.append([0.02, 5, 0.6815, 0.701, 0, 6, 4, 0])     #0.84246878      

models.append([0.005, 5, 0.6815, 0.701, 0, 6, 1, 0])
'''

n_iterations_outer = 12
n_folds_outer = 10

n_iterations_inner = 5
n_folds_inner = 10

np.random.seed(4242)

def func_cv_2(X, y, folds, model, verbose):
    scores = []
    for train, test in folds:
        print('**func_cv_2 Fold', 1 + len(scores), 'of', n_folds_outer)
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        #todo: pass folds to predict?
        y_predicted = np.zeros(len(y_test))
        for i in range(len(models)):
            y_predicted_per_model = my_predict(X_train, y_train, X_test, n_iterations_inner, n_folds_inner, func_predict_1, models[i], verbose=False)
            score_per_model = roc_auc_score(y_test, y_predicted_per_model)
            print('model', i, score_per_model, models[i])

            if i == 0:
                y_predicted = y_predicted_per_model
            elif i <= 3:
                y_predicted += 1/3*y_predicted_per_model
            else:
                foo += 1

        if len(models) == 4:
            y_predicted = y_predicted/2
        else:
            foo += 1

        score = roc_auc_score(y_test, y_predicted)
        print('**func_cv_2 auc score for combined model', score)
        scores.append(score)

    scores = np.array(scores)
    print('****func_cv_2: mean, std', scores.mean(), scores.std())
    return scores.mean()


my_cv(X, y, n_iterations_outer, n_folds_outer, func_cv_2)

