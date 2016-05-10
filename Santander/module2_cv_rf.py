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


values_n_estimators = [5000]
values_max_depth = [5,7,9]

iterations = 5
n_folds = 5

for n_estimators in values_n_estimators:
    for max_depth in values_max_depth:
		#seed np.random so the kfolds are identical in each iteration 
		#randomness in kfold. no random seed set in clf
        np.random.seed(4242)

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced_subsample', max_features = None, n_jobs = 8)

        results = []
        for i in range(iterations):
            kfolds = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
            results_per_iteration = []
            for train, test in kfolds:
                X_train, y_train = X[train], y[train]
                X_test, y_test = X[test], y[test]

                clf.fit(csr_matrix(X_train), y_train)
                y_predicted = clf.predict_proba(csr_matrix(X_test))
                roc = roc_auc_score(y_test, y_predicted[:, 1])
                print('roc', roc)
                results_per_iteration.append(roc)

            print('roc mean, std', np.array(results_per_iteration).mean(), np.array(results_per_iteration).std())
            results.append(np.array(results_per_iteration).mean())

        print('mean, std', np.array(results).mean(), np.array(results).std())
        a = np.array(results);
        #a = np.delete(a, [a.argmax(), a.argmin()])
        print(a.mean(), '\t', a.std(), '\t', n_estimators, max_depth)

print('Jobs done', datetime.now())


