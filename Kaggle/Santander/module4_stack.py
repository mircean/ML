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

xgb = 1
rf_ab = 0

models = []

'''
values_learning_rate = [0.01,0.02]		#default 0.1
values_max_depth = [4,5,6]				#default 6
values_ss = [0.6815]			#default 1
values_cs = [0.701]			#default 1
values_gamma = [0]				#default 0
values_min_child_weight = [1,3,6]	#default 1
values_lambda = [1,2]			#default 1
values_alpha = [0]				#default 0

for learning_rate in values_learning_rate:
    for max_depth in values_max_depth:
        for ss in values_ss:
            for cs in values_cs:
                for gamma in values_gamma:
                    for min_child_weight in values_min_child_weight:
                        for reg_lambda in values_lambda:
                            for reg_alpha in values_alpha:
                                models.append([learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha])
'''

#models.append([0.0202064, 5, 0.6815, 0.701, 0, 1, 1, 0])
#models.append([0.03, 5, 0.8, 0.7, 0, 1, 1, 0])
models.append([0.02, 5, 0.6815, 0.701, 0, 6, 1, 0])

def predictRFAB(is_rf, X, y, X_predict, n_iterations):
    for i in range(n_iterations):
        if is_rf == 1:
            clf = RandomForestClassifier(n_estimators=500, max_depth=7, class_weight='balanced_subsample', max_features=None, n_jobs=8, verbose=False)
        else:
            clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.3, random_state=np.random)
            #todo: why doesn't  random state matter and predict returns the same exact numbers in each iteration?

        clf.fit(csr_matrix(X), y)
        #y_predicted = clf.predict_proba(csr_matrix(X_predict))[:, 1]
        y_predicted = clf.predict(csr_matrix(X_predict))
        print(y_predicted)

        if i == 0:
            scores = y_predicted
        else:
            #scores *= y_predicted
            scores += y_predicted

    #scores = np.power(scores, 1./iterations)
    return scores

df_train2 = pd.DataFrame()
df_test2 = pd.DataFrame()

n_iterations_outer = 1
n_folds_outer = 25

n_iterations_inner = 5
n_folds_inner = 10

if xgb == 1:
    i = 0
    for x in models:
        np.random.seed(4242)

        new_feature_train = np.ones(len(y))
        for j in range(n_iterations_outer):
            print('Iteration', j + 1, 'of', n_iterations_outer)

            new_feature_train_per_iteration = np.zeros(len(y))

            folds = StratifiedKFold(y, n_folds=n_folds_outer, shuffle=True)
            for train, test in folds:
                print('fold')
                
                X_train, y_train = X[train], y[train]
                X_test, y_test = X[test], y[test]

                y_predicted = my_predict(X_train, y_train, X_test, n_iterations_inner, n_folds_inner, func_predict_1, models[i], verbose=False)
                score = roc_auc_score(y_test, y_predicted)
                print('  score per fold', score)

                for k in range(len(test)):
                    new_feature_train_per_iteration[test[k]] = y_predicted[k]

            score = roc_auc_score(y, new_feature_train_per_iteration)
            print(' score per iteration', score)

            new_feature_train *= new_feature_train_per_iteration
           
        new_feature_train = np.power(new_feature_train, 1./n_iterations_outer)
        score = roc_auc_score(y, new_feature_train)
        print('score for new feature', score)

        new_feature_test = my_predict(X, y, X_predict, n_iterations_inner, n_folds_inner, func_predict_1, models[i], verbose=False)

        df_train2['column' + str(i)] = new_feature_train
        df_test2['column' + str(i)] = new_feature_test

        i += 1
        #if i == 2:
        #    break

if rf_ab == 1:
    np.random.seed(4242)

    halfs = StratifiedKFold(y, n_folds=2, shuffle=True)
    for x1, x2 in halfs:
        half1 = x1
        half2 = x2
        break

    X1, y1 = X[half1], y[half1]
    X2, y2 = X[half2], y[half2]

    predict1 = predictRFAB(1, X1, y1, X2, n_iterations_inner)
    print('half1 roc score', roc_auc_score(y2, predict1))
    predict2 = predictRFAB(1, X2, y2, X1, n_iterations_inner)
    print('half2 roc score', roc_auc_score(y1, predict2))
    predict3 = predictRFAB(1, X, y, X_predict, n_iterations_inner)

    predict4=np.zeros(len(y))
    for j in range(len(half1)):
        predict4[half1[j]] = predict2[j]
    for j in range(len(half2)):
        predict4[half2[j]] = predict1[j]
    
    df_train2['column0'] = predict4
    df_test2['column0'] = predict3

    predict1 = predictRFAB(0, X1, y1, X2, n_iterations_inner)
    print('half1 roc score', roc_auc_score(y2, predict1))
    predict2 = predictRFAB(0, X2, y2, X1, n_iterations_inner)
    print('half2 roc score', roc_auc_score(y1, predict2))
    predict3 = predictRFAB(0, X, y, X_predict, n_iterations_inner)

    predict4=np.zeros(len(y))
    for j in range(len(half1)):
        predict4[half1[j]] = predict2[j]
    for j in range(len(half2)):
        predict4[half2[j]] = predict1[j]
    
    df_train2['column1'] = predict4
    df_test2['column1'] = predict3


df_train2.to_csv('Output\\Stacking_Train.csv', index=False)
df_test2.to_csv('Output\\Stacking_Test.csv', index=False)

print('done', datetime.now())