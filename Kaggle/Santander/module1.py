import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.core import DMatrix
from xgboost.training import train, cv
from xgboost.plotting import plot_importance

from feature_engineering import feature_engineering
from cv_common import my_cv, my_predict, func_cv_1, func_predict_1

#seed(0) is implicit; another seed can make a significant difference
np.random.seed(0)

#Loading data
print('load data', datetime.now())
df_train = pd.read_csv('Dataset\\train.csv')
print('train data', df_train.shape) 
df_test = pd.read_csv('Dataset\\test.csv')
print('test data', df_test.shape) 

labels = df_train['TARGET'].values
df_train = df_train.drop(['TARGET'], axis=1)
id_test = df_test['ID']
piv_train = df_train.shape[0]

#Creating a DataFrame with train+test data
#df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

#Feature engineering
#df_all = feature_engineering(df_all, piv_train)
#df_all.to_csv('Dataset\\df_all.csv', index=False)
df_all = pd.read_csv('Dataset\\df_all.csv')


#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
y = labels
X_predict = vals[piv_train:]


test_xgb = 0
test_rf = 0
test_ab = 0
test_force0 = 0
cv_0 = 0
cv_1 = 0
cv_1_rf = 0
cv_1_ab = 0
cv_2 = 0
cv_3 = 0
submit = 0
submit_1 = 0
submit_1_rf = 0
submit_1_ab = 0
submit_2 = 0
submit_21 = 0
submit_22 = 0
submit_3 = 0

force0 = 0

if sys.argv[1] == 'test':
    if sys.argv[2] == 'xgb':
        test_xgb = 1
    if sys.argv[2] == 'rf':
        test_rf = 1
    if sys.argv[2] == 'ab':
        test_ab = 1
if sys.argv[1] == 'cv':
    if sys.argv[2] == '1':
        cv_1 = 1
    if sys.argv[2] == '1_rf':
        cv_1_rf = 1
    if sys.argv[2] == '1_ab':
        cv_1_ab = 1
    if sys.argv[2] == '3':
        cv_3 = 1
if sys.argv[1] == 'submit':
    if sys.argv[2] == '1':
        submit_1 = 1
    if sys.argv[2] == '1_rf':
        submit_1_rf = 1
    if sys.argv[2] == '1_ab':
        submit_1_ab = 1
if sys.argv[1] == 'test_force0':
    test_force0 = 1


#first try
#learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 6, 0.5, 0.5, 0, 1, 1, 0 

#tuning#1 using airbnb method
#learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.01, 6, 0.75, 0.5, 0, 2, 2, 1
#try 0.02
learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.02, 6, 0.75, 0.5, 0, 2, 2, 1

#tuning#2
#learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 6, 0.7, 0.7, 0.5, 10, 1, 0
#learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 5, 0.75, 0.6, 0, 8, 1, 0

#from forum
#learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.03, 5, 0.8, 0.7, 0, 1, 1, 0
#learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.0202064, 5, 0.6815, 0.701, 0, 1, 1, 0

#tuning3
#learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.02, 5, 0.6815, 0.701, 0, 6, 1, 0

early_stopping_rounds = 100
if learning_rate == 0.1:
    early_stopping_rounds = 15
if learning_rate == 0.03:
    early_stopping_rounds = 50
if learning_rate == 0.01:
    early_stopping_rounds = 100

print(learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha, early_stopping_rounds)

if test_xgb == 1:
    #randomness in train_test_split. no random seed set in clf
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='binary:logistic', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)      
    clf.fit(csr_matrix(X_train), y_train, eval_set=[(csr_matrix(X_train), y_train), (csr_matrix(X_test), y_test)], eval_metric='auc', early_stopping_rounds=early_stopping_rounds, verbose=10)
    #y_predicted = clf.predict(csr_matrix(X_test), ntree_limit=clf.booster().best_ntree_limit)
    y_predicted = clf.predict_proba(csr_matrix(X_test), ntree_limit=clf.booster().best_ntree_limit)
    print('roc score', roc_auc_score(y_test, y_predicted[:, 1]))

    plot_importance(clf.booster())
    plt.show()
    ''' 
	top 10 columns
	'var38'
	'var15'
	'saldo_var30'
	'saldo_medio_var5_hace3'
	'saldo_medio_var5_hace2'
	'saldo_medio_var5_ult3'
	'num_var22_ult3'
	'saldo_medio_var5_ult1'
	'saldo_var42'
	'num_var45_hace3'
	'''
    plot = 1
    if plot == 1:
        fpr, tpr, _ = roc_curve(y_test, y_predicted[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.6f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

if test_rf == 1:
    #randomness in train_test_split. no random seed set in clf
    for i in range(3):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #clf = RandomForestClassifier(n_estimators=1000, max_depth=5, class_weight='balanced', n_jobs = 8)
        clf = RandomForestClassifier(n_estimators=1000, max_depth=5, class_weight='balanced_subsample', max_features = 'auto', n_jobs = 8, verbose = False)
        clf.fit(csr_matrix(X_train), y_train)
        y_predicted = clf.predict_proba(csr_matrix(X_test))
        print('roc score', roc_auc_score(y_test, y_predicted[:, 1]))

        print(clf.predict_proba(X_predict))

    np.random.seed(0)
    for i in range(3):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #clf = RandomForestClassifier(n_estimators=1000, max_depth=5, class_weight='balanced', n_jobs = 8)
        clf = RandomForestClassifier(n_estimators=1000, max_depth=5, class_weight='balanced_subsample', max_features = None, n_jobs = 8, verbose = False)
        clf.fit(csr_matrix(X_train), y_train)
        y_predicted = clf.predict_proba(csr_matrix(X_test))
        print('roc score', roc_auc_score(y_test, y_predicted[:, 1]))

        print(clf.predict_proba(X_predict))

if test_ab == 1:
    #randomness in train_test_split. no random seed set in clf
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
        clf.fit(csr_matrix(X_train), y_train)
        y_predicted = clf.predict_proba(csr_matrix(X_test))
        print('roc score', roc_auc_score(y_test, y_predicted[:, 1]))
        print(y_predicted[:, 1])

        y_predicted = clf.predict_proba(csr_matrix(X_predict))
        print(y_predicted[:, 1])

#very bad idea
if test_force0 == 1:
    kfolds = StratifiedKFold(y, n_folds=5, shuffle=True)
    for x1,x2 in kfolds:
        train,test=x1,x2
        break
    
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='binary:logistic', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)      
    clf.fit(csr_matrix(X_train), y_train, eval_set=[(csr_matrix(X_train), y_train), (csr_matrix(X_test), y_test)], eval_metric='auc', early_stopping_rounds=early_stopping_rounds, verbose=100)
    y_predicted = clf.predict_proba(csr_matrix(X_test), ntree_limit=clf.booster().best_ntree_limit)
    scores = y_predicted[:, 1]

    print(scores[:20])
    print('roc score', roc_auc_score(y_test, scores))

    num_var33 = df_train['num_var33'] + df_train['saldo_medio_var33_ult3'] + df_train['saldo_medio_var44_hace2'] + df_train['saldo_medio_var44_hace3'] + df_train['saldo_medio_var33_ult1'] + df_train['saldo_medio_var44_ult1']
    scores[num_var33[test].values > 0] = 0
    scores[df_train['var15'][test].values < 23] = 0
    scores[df_train['saldo_medio_var5_hace2'][test].values > 160000] = 0
    scores[df_train['saldo_var33'][test].values > 0] = 0
    scores[df_train['var38'][test].values > 3988596] = 0
    scores[df_train['var21'][test].values > 7500] = 0
    scores[df_train['num_var30'][test].values > 9] = 0
    scores[df_train['num_var13_0'][test].values > 6] = 0
    scores[df_train['num_var33_0'][test].values > 0] = 0
    scores[df_train['imp_ent_var16_ult1'][test].values > 51003] = 0
    scores[df_train['imp_op_var39_comer_ult3'][test].values > 13184] = 0
    scores[df_train['saldo_medio_var5_ult3'][test].values > 108251] = 0
    
    print(scores[:20])
    print('roc score', roc_auc_score(y_test, scores))

    scores[df_train['num_var37_0'][test].values > 45] = 0
    scores[df_train['saldo_var5'][test].values > 137615] = 0
    scores[df_train['saldo_var8'][test].values > 60099] = 0
    var15 = df_train['var15'] + df_train['num_var45_hace3'] + df_train['num_var45_ult3'] + df_train['var36']
    scores[var15[test].values <= 24] = 0
    scores[df_train['saldo_var14'][test].values > 19053.78] = 0
    scores[df_train['saldo_var17'][test].values > 288188.97] = 0
    scores[df_train['saldo_var26'][test].values > 10381.29] = 0    
    
    print(scores[:20])
    print('roc score', roc_auc_score(y_test, scores))

if cv_0 == 1:
    #UPDATE, use sparse!!!
	#randomness in train_test_split. no random seed set in clf
	iterations  = 25
	train_and_test_scores = np.zeros(iterations)

	for i in range(iterations):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		
		clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='binary:logistic', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)      
		clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=early_stopping_rounds, verbose=False)
		score = clf.booster().best_score
		print(score, clf.booster().best_ntree_limit)
		train_and_test_scores[i] = score

	print(train_and_test_scores)
	mean = round(100*np.array(train_and_test_scores).mean(), 4)
	std = round(100*np.array(train_and_test_scores).std(), 4)
	stderr = round(std/sqrt(iterations), 4)

	print('mean score', mean, 'std', std, 'stderr', stderr)
	for x in train_and_test_scores:
		print(x)

#best so far. the seed doesn't matter much (low variance) and is correct (low bias)
#randomness in kfold. no random seed set in clf
if cv_1 == 1:
    #np.random.seed(0)
    #np.random.seed(1234)
    np.random.seed(4242)

    n_iterations = 12
    n_folds = 10
    my_cv(X, y, n_iterations, n_folds, func_cv_1, [learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha])

if cv_1_rf == 1 or cv_1_ab == 1:
    #np.random.seed(0)
    #np.random.seed(4242)
    np.random.seed(1234)

    #randomness in kfold. no random seed set in clf
    if cv_1_rf == 1:
        clf = RandomForestClassifier(n_estimators=1000, max_depth=5, class_weight='balanced_subsample', max_features = None, n_jobs = 8)
    else:
        clf = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)    

    iterations = 5
    n_folds = 5

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
            results_per_iteration.append(roc)

        print('roc mean, std', np.array(results_per_iteration).mean(), np.array(results_per_iteration).std())
        results.append(np.array(results_per_iteration).mean())

    print('mean, std', np.array(results).mean(), np.array(results).std())
    a = np.array(results);
    #a = np.delete(a, [a.argmax(), a.argmin()])
    print('mean2', 'std2', a.mean(), a.std())

#not a good solution. low std but high bias
if cv_2 == 1:
    #UPDATE, use sparse!!!
	#randomness in clf. no random seed set in kfold. 
	iterations = 5

	kfolds = StratifiedKFold(y, n_folds=10, shuffle=True)
	np.random.seed(0)

	results = []
	n_estimators = []
	for i in range(iterations):
		clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='binary:logistic', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha, seed=np.random.randint(1024))      

		xgb_options = clf.get_xgb_params()
		xgb_options.update({"eval_metric":'auc'})
		train_dmatrix = DMatrix(X, label=y)

		cv_results = cv(xgb_options, train_dmatrix, clf.n_estimators, early_stopping_rounds=early_stopping_rounds, maximize=True, show_progress=False, show_stdv=False, folds=kfolds)
		print('mean, std, iterations', cv_results.values[-1][0], cv_results.values[-1][1], cv_results.shape[0])
		results.append(cv_results.values[-1][0])
		n_estimators.append(cv_results.shape[0])

	print('mean, std, iterations', np.array(results).mean(), np.array(results).std(), np.array(n_estimators).mean())

if cv_3 == 1:
    n_iterations = 12
    n_folds = 10

    def func_cv_local(X, y, folds, model, verbose, seed):
        score = my_cv(X, y, n_iterations, n_folds, func_cv_1, [learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha])
        return score

    my_cv(X, y, n_iterations, n_folds, func_cv_local, [learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha])

#weakness - same number of trees for all models
#weakness - not much variance when changing the clf random seed (see cv_2, too)
if submit == 1:
    #randomness in train_test_split. no random seed set in clf
    iterations = 10
    n_estimators = 1200

    if learning_rate == 0.1:
        n_estimators = 200
    if learning_rate == 0.03:
        n_estimators = 500

    scores = np.zeros(n_estimators)

    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective='binary:logistic', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)      
        clf.fit(csr_matrix(X_train), y_train, eval_set=[(csr_matrix(X_test), y_test)], eval_metric='auc', verbose=False)

        scores_crt = np.array([float(x) for x in clf.evals_result()['validation_0']['auc']])
        print('best score', scores_crt.max(), scores_crt.argmax())
        scores = scores + scores_crt

    print(scores.max(), scores.argmax())
    #plt.plot(scores)
    #plt.show()

    n_estimators = scores.argmax()

    #randomness in clf, need different seed per iteration
    #Fixed number of n_estimators

    scores = np.zeros(X_predict.shape[0])

    for i in range(iterations):
        print('Iteration', i)
        clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective='binary:logistic', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha, seed=np.random.randint(1024))      
        clf.fit(csr_matrix(X), y)
        y_predicted = clf.predict_proba(csr_matrix(X_predict))
        scores = scores + y_predicted[:, 1]

    scores = scores / iterations;
    print(scores)

#similar to cv_1
#create 12 modelds, train each one, find best number of trees in cv, fit each model. a bit better than sumbit
if submit_1 == 1:
    #np.random.seed(0)
    np.random.seed(1234)
    #np.random.seed(4242)
    #randomness in kfold. no random seed set in clf

    n_iterations = 12
    n_folds = 10

    scores = my_predict(X, y, X_predict, n_iterations, n_folds, func_predict_1, [learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha])
    print(scores)

if submit_1_rf == 1 or submit_1_ab == 1:
    #np.random.seed(0)
    np.random.seed(4242)

    #randomness in kfold. no random seed set in clf

    iterations = 5
    for i in range(iterations):
        if submit_1_rf == 1:
            clf = RandomForestClassifier(n_estimators=500, max_depth=7, class_weight='balanced_subsample', max_features = None, n_jobs = 8)
        else:
            clf = AdaBoostClassifier(n_estimators=500, learning_rate=0.1, random_state=np.random)
        
        clf.fit(csr_matrix(X), y)
        y_predicted = clf.predict_proba(csr_matrix(X_predict))
        print(y_predicted[:, 1])

        if i == 0:
            scores = y_predicted[:, 1]
            scores2 = np.copy(scores)
        else:
            scores *= y_predicted[:, 1]
            scores2 += y_predicted[:, 1]

    scores = np.power(scores, 1./iterations)
    print(scores)
    scores2 = scores2/iterations
    print(scores2)

if submit_2 == 1:
    #randomness in kfold. no random seed set in clf
    #faster than submit_1. potential issue - variance if changing the random seed which changes the folds
    clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='binary:logistic', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)      

    n_folds = 10
    kfolds = StratifiedKFold(y, n_folds=n_folds, shuffle=True)

    scores = None

    for train, test in kfolds:
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        clf.fit(csr_matrix(X_train), y_train, eval_set=[(csr_matrix(X_test), y_test)], eval_metric='auc', early_stopping_rounds=early_stopping_rounds, verbose=False)
        y_predicted = clf.predict_proba(csr_matrix(X_test), ntree_limit=clf.booster().best_ntree_limit)

        print('roc score', roc_auc_score(y_test, y_predicted[:, 1]), 'score', clf.booster().best_score, 'trees', clf.booster().best_ntree_limit)	

        y_predicted = clf.predict_proba(csr_matrix(X_predict), ntree_limit=clf.booster().best_ntree_limit)
        if scores is None:
            scores = y_predicted[:, 1]
        else:
            scores *= y_predicted[:, 1]

    print('done model', datetime.now())
    scores = np.power(scores, 1./n_folds)
    print(scores)

if submit_21 == 1:
    #virtually identical with script in forum
    num_rounds = 350
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.03
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.7
    params["silent"] = 1
    params["max_depth"] = 5
    params["min_child_weight"] = 1
    params["eval_metric"] = "auc"

    n_folds = 10
    kfolds = StratifiedKFold(y,n_folds=n_folds, shuffle=False, random_state=42)
    scores = None

    for train, test in kfolds:
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]

        dtrain = xgb.DMatrix(csr_matrix(X_train), y_train, silent=True)
        dtest = xgb.DMatrix(csr_matrix(X_test), y_test, silent=True)
        clf = xgb.train(params, dtrain, num_rounds)
        y_predicted = clf.predict(dtest)
        print('roc score', roc_auc_score(y_test, y_predicted))

        dpredict = xgb.DMatrix(csr_matrix(X_predict), silent=True)
        y_predicted = clf.predict(dpredict)
        
        if scores is None:
            scores = y_predicted
        else:
            scores *= y_predicted

    scores = np.power(scores, 1./n_folds)
    print(scores)

    sub = pd.DataFrame(np.column_stack((id_test, scores)), columns=['ID', 'TARGET'])
    sub['ID'] = sub['ID'].astype(int)

    filename = 'results.csv'
    sub.to_csv(filename, index=False)

if submit_22 == 1:
    #virtually identical with script in forum
    np.random.seed(1234)
    num_rounds = 560
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.0202064
    params["subsample"] = 0.6815
    params["colsample_bytree"] = 0.701
    params["max_depth"] = 5
    params["eval_metric"] = "auc"


    dtrain = xgb.DMatrix(csr_matrix(X), y)
    dtest = xgb.DMatrix(csr_matrix(X_predict))
    evals = [(dtrain, 'train')]
    clf = xgb.train(params, dtrain, num_rounds, evals=evals)
    scores = clf.predict(dtest)
        
#very slow, do not use
if submit_3 == 1:
    #UPDATE, use sparse!!!
	#randomness in clf and in train_test_split

	models = 10
	iterations = 10

	n_estimators = 1200
	if learning_rate == 0.1:
		n_estimators = 200
	if learning_rate == 0.03:
		n_estimators = 500

	scores = np.zeros(X_predict.shape[0])
	for i in range(models):
		scores_inner = np.zeros(n_estimators)
		seed=np.random.randint(1024)
		clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective='binary:logistic', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha, seed=seed)      
		for j in range(iterations):
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
			clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', verbose=False)

			scores_crt = np.array([float(x) for x in clf.evals_result()['validation_0']['auc']])
			print('best score', scores_crt.max(), scores_crt.argmax())
			scores_inner = scores_inner + scores_crt

		print(scores_inner.max(), scores_inner.argmax())
		n_estimators2 = scores_inner.argmax()

		clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators2, objective='binary:logistic', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha, seed=seed)      
		clf.fit(X, y)
		y_predicted = clf.predict_proba(X_predict)
		print(y_predicted[:, 1])
		scores = scores + y_predicted[:, 1]

	scores = scores / models
	print(scores)

if submit == 1 or submit_1 == 1 or submit_2 == 1 or submit_21 == 1 or submit_22 == 1 or submit_3 == 1 or submit_1_rf or submit_1_ab:
    if force0 == 1:
        num_var33 = df_test['num_var33'] + df_test['saldo_medio_var33_ult3'] + df_test['saldo_medio_var44_hace2'] + df_test['saldo_medio_var44_hace3'] + df_test['saldo_medio_var33_ult1'] + df_test['saldo_medio_var44_ult1']
        scores[num_var33.values > 0] = 0
        scores[df_test['var15'].values < 23] = 0
        scores[df_test['saldo_medio_var5_hace2'].values > 160000] = 0
        scores[df_test['saldo_var33'].values > 0] = 0
        scores[df_test['var38'].values > 3988596] = 0
        scores[df_test['var21'].values > 7500] = 0
        scores[df_test['num_var30'].values > 9] = 0
        scores[df_test['num_var13_0'].values > 6] = 0
        scores[df_test['num_var33_0'] .values> 0] = 0
        scores[df_test['imp_ent_var16_ult1'].values > 51003] = 0
        scores[df_test['imp_op_var39_comer_ult3'].values > 13184] = 0
        scores[df_test['saldo_medio_var5_ult3'] .values> 108251] = 0

        scores[df_test['num_var37_0'].values > 45] = 0
        scores[df_test['saldo_var5'].values > 137615] = 0
        scores[df_test['saldo_var8'].values > 60099] = 0
        var15 = df_test['var15'] + df_test['num_var45_hace3'] + df_test['num_var45_ult3'] + df_test['var36']
        scores[var15.values <= 24] = 0
        scores[df_test['saldo_var14'].values > 19053.78] = 0
        scores[df_test['saldo_var17'].values > 288188.97] = 0
        scores[df_test['saldo_var26'].values > 10381.29] = 0    

    sub = pd.DataFrame(np.column_stack((id_test, scores)), columns=['ID', 'TARGET'])
    sub['ID'] = sub['ID'].astype(int)

    filename = 'Output\\results.csv'
    sub.to_csv(filename, index=False)

print('done', datetime.now())


'''
#df_1 = pd.read_csv('Ensemble\\results0_840916.csv')
#df_1 = pd.read_csv('Ensemble\\results1_840610.csv')
#df_1 = pd.read_csv('Ensemble\\results2_841266_noforce0.csv')
#df_1 = pd.read_csv('Ensemble\\scirpus_841089.csv')
#df_1 = pd.read_csv('Ensemble\\xgb_lalala_841664_noforce0.csv')
df_1 = pd.read_csv('Ensemble2\\results00-0 - apply force0 again.csv')
df_test = pd.read_csv('Dataset\\test.csv')

#for x in df_test.columns:
#    min1 = df_test[x].min()
#    max1 = df_test[x].max()
#    df_test[x][df_test[x] < min1] = min1
#    df_test[x][df_test[x] > max1] = max1

scores = df_1.TARGET.values

num_var33 = df_test['num_var33'] + df_test['saldo_medio_var33_ult3'] + df_test['saldo_medio_var44_hace2'] + df_test['saldo_medio_var44_hace3'] + df_test['saldo_medio_var33_ult1'] + df_test['saldo_medio_var44_ult1']
scores[num_var33.values > 0] = 0
scores[df_test['var15'].values < 23] = 0
scores[df_test['saldo_medio_var5_hace2'].values > 160000] = 0
scores[df_test['saldo_var33'].values > 0] = 0
scores[df_test['var38'].values > 3988596] = 0
scores[df_test['var21'].values > 7500] = 0
scores[df_test['num_var30'].values > 9] = 0
scores[df_test['num_var13_0'].values > 6] = 0
scores[df_test['num_var33_0'].values > 0] = 0
scores[df_test['imp_ent_var16_ult1'].values > 51003] = 0
scores[df_test['imp_op_var39_comer_ult3'].values > 13184] = 0
scores[df_test['saldo_medio_var5_ult3'].values > 108251] = 0

scores[df_test['num_var37_0'].values > 45] = 0
scores[df_test['saldo_var5'].values > 137615] = 0
scores[df_test['saldo_var8'].values > 60099] = 0
var15 = df_test['var15'] + df_test['num_var45_hace3'] + df_test['num_var45_ult3'] + df_test['var36']
scores[var15.values <= 24] = 0
scores[df_test['saldo_var14'].values > 19053.78] = 0
scores[df_test['saldo_var17'].values > 288188.97] = 0
scores[df_test['saldo_var26'].values > 10381.29] = 0 

#df_1['TARGET'] = scores
df_1.to_csv('results_force0.csv', index=False)
'''

