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

values_learning_rate = [0.02]		#default 0.1
values_max_depth = [5]				#default 6
values_ss = [0.6815]			#default 1
values_cs = [0.701]			#default 1
values_gamma = [0]				#default 0
values_min_child_weight = [6]	#default 1
values_lambda = [1,2,4,8,16]				#default 1
values_alpha = [0]				#default 0

iterations = 12
n_folds = 10

for learning_rate in values_learning_rate:
    for max_depth in values_max_depth:
        for ss in values_ss:
            for cs in values_cs:
                for gamma in values_gamma:
                    for min_child_weight in values_min_child_weight:
                        for reg_lambda in values_lambda:
                            for reg_alpha in values_alpha:
								#seed np.random so the kfolds are identical in each iteration 
								#randomness in kfold. no random seed set in clf
                                np.random.seed(4242)

                                n_iterations = 12
                                n_folds = 10
                                score = my_cv(X, y, n_iterations, n_folds, func_cv_1, [learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha])

                                #TODO: need std, too
                                print(score, '\t', learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha)

print('Jobs done', datetime.now())


