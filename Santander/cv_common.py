import numpy as np

from scipy.sparse import csr_matrix

from sklearn.cross_validation import StratifiedKFold

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.core import DMatrix
from xgboost.training import train, cv

def func_cv_1(X, y, folds, model, verbose):
    learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = model[0], model[1], model[2], model[3], model[4], model[5], model[6], model[7]
    early_stopping_rounds = 100
    if learning_rate == 0.1:
        early_stopping_rounds = 15
    if learning_rate == 0.03:
        early_stopping_rounds = 50
    if learning_rate == 0.01:
        early_stopping_rounds = 100

    clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='binary:logistic', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)      

    xgb_options = clf.get_xgb_params()
    xgb_options.update({"eval_metric":'auc'})
    train_dmatrix = DMatrix(csr_matrix(X), label=y)

    cv_results = cv(xgb_options, train_dmatrix, clf.n_estimators, early_stopping_rounds=early_stopping_rounds, maximize=True, verbose_eval=False, show_stdv=False, folds=folds)
    if verbose == True:
        print('cv1: mean, std, iterations', cv_results.values[-1][0], cv_results.values[-1][1], cv_results.shape[0])

    return cv_results.values[-1][0]

def func_predict_1(X, y, X_predict, folds, model, verbose):
    learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = model[0], model[1], model[2], model[3], model[4], model[5], model[6], model[7]
    early_stopping_rounds = 100
    if learning_rate == 0.1:
        early_stopping_rounds = 15
    if learning_rate == 0.03:
        early_stopping_rounds = 50
    if learning_rate == 0.01:
        early_stopping_rounds = 100

    clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='binary:logistic', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)      

    xgb_options = clf.get_xgb_params()
    xgb_options.update({"eval_metric":'auc'})
    train_dmatrix = DMatrix(csr_matrix(X), label=y)

    cv_results = cv(xgb_options, train_dmatrix, clf.n_estimators, early_stopping_rounds=early_stopping_rounds, maximize=True, verbose_eval=False, show_stdv=False, folds=folds)
    if verbose == True:
        print('predict1: mean, std, iterations', cv_results.values[-1][0], cv_results.values[-1][1], cv_results.shape[0])

    n_estimators = cv_results.shape[0]
    clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective='binary:logistic', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)      
    clf.fit(csr_matrix(X), y)
    y_predicted = clf.predict_proba(csr_matrix(X_predict))
    return y_predicted[:, 1]

#randomness in kfold. no random seed set in clf
#best CV so far. the seed doesn't matter much (low variance) and is correct (low bias)
def my_cv(X, y, n_iterations, n_folds, func_cv, model=None, verbose=True):
    scores = []
    for i in range(n_iterations):
        if verbose == True:
            print('my_cv iteration', i + 1, 'of', n_iterations)

        folds = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
        score = func_cv(X, y, folds, model, verbose)
        scores.append(score)

    scores = np.array(scores)
    scores = np.delete(scores, [scores.argmax(), scores.argmin()])
    if verbose == True:
        print('my_cv mean, std', scores.mean(), scores.std())

    return scores.mean()

def my_predict(X, y, X_predict, n_iterations, n_folds, func_predict, model, average='Geometric', verbose=True):
    if average == 'Geometric':
        y_predicted = np.ones(X_predict.shape[0])
    else:
        y_predicted = np.zeros(X_predict.shape[0])

    for i in range(n_iterations):
        if verbose == True:
            print('my_predict iteration', i + 1, 'of', n_iterations)
        
        folds = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
        y_predicted_per_iteration = func_predict(X, y, X_predict, folds, model, verbose)

        if average == 'Geometric':
            y_predicted *= y_predicted_per_iteration
        else:
            y_predicted += y_predicted_per_iteration

    if average == 'Geometric':
         y_predicted = np.power(y_predicted, 1./n_iterations)
    else:
        y_predicted /= n_iterations
    
    return y_predicted
