import datetime
from datetime import datetime
import math
from math import sqrt
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

def calculate_season(x):
	if x.month in (3,4,5): 
		return 0
	if x.month in (6,7,8):
		return 1
	if x.month in (9,10,11):
		return 2
	return 3

jobs = []
thread_count = 8
iterations_per_job = 10

class myThread (threading.Thread):
	def __init__(self, threadID):
		threading.Thread.__init__(self)
		self.threadID = threadID

	def run(self):
		iterations = math.ceil(len(jobs)/thread_count)
		print(self.threadID, 'Starting jobs', iterations)
		for i in range(iterations):
			job_index = self.threadID*iterations + i
			if job_index < len(jobs):
				job = jobs[job_index]
				job[2] = job[0](job[1])
				#print('Thread', self.threadID, job[2])
		#print(self.threadID, 'Done')

def calculate_score_2(y_predicted, y_true):
	score = calculate_score(y_predicted, y_true.get_label())
	score = 1 - score
	return 'myscore', score

def calculate_score_per_row(y_predicted, y_true):
	top5 = np.argsort(y_predicted)[::-1][:5]
	score = 0
	for i in range(5):
		if(top5[i] == y_true):
			score = score + 1/math.log2(i+2)
			break
	return score

def calculate_score(y_predicted, y_true):
	scores = [calculate_score_per_row(x[0], x[1]) for x in zip(y_predicted, y_true)]
	score = np.array(scores).mean()
	return score

np.random.seed(0)

#dataset_no = 0 #w/o sessions, latest features
dataset_no = 1 #sessions, drop !hassessions, no latest features
#dataset_no = 2 #sessions, drop !hassessions, latest features

if dataset_no == 0:
	#Loading data
	print('load data', datetime.now())

	df_train = pd.read_csv('Dataset\\train_users_2.csv')
	print('rows', df_train.shape[0]) 

	df_test = pd.read_csv('Dataset\\test_users.csv')
	labels = df_train['country_destination'].values
	df_train = df_train.drop(['country_destination'], axis=1)
	id_test = df_test['id']
	piv_train = df_train.shape[0]

	#Creating a DataFrame with train+test data
	df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

	#Removing id and date_first_booking
	df_all = df_all.drop(['id', 'date_first_booking'], axis=1)

	#Filling nan
	df_all = df_all.fillna(-1)

	#####Feature engineering#######
	print('features in the csv', df_all.shape[1])

	#date_account_created
	print('dac', datetime.now())
	dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
	df_all['dac_year'] = dac[:,0]
	df_all['dac_month'] = dac[:,1]
	df_all['dac_day'] = dac[:,2]

	#day of week, seazon
	print('dac2', datetime.now())
	dac2 = df_all.date_account_created.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
	df_all['dac_weekday'] = dac2.apply(lambda x: x.weekday())
	df_all['dac_season'] = dac2.apply(calculate_season)

	df_all = df_all.drop(['date_account_created'], axis=1)

	#timestamp_first_active
	print('tfa', datetime.now())
	tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
	df_all['tfa_year'] = tfa[:,0]
	df_all['tfa_month'] = tfa[:,1]
	df_all['tfa_day'] = tfa[:,2]
	df_all['tfa_hour'] = tfa[:,3]

	#day of week, season
	print('tfa2', datetime.now())
	tfa2 = df_all.timestamp_first_active.astype(str).apply(lambda x: datetime.strptime(x[:8], '%Y%m%d'))
	df_all['tfa_weekday'] = tfa2.apply(lambda x: x.weekday())
	df_all['tfa_season'] = tfa2.apply(calculate_season)

	df_all = df_all.drop(['timestamp_first_active'], axis=1)

	#Age
	print('age', datetime.now())
	av = df_all.age.values
	df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

	print('features in the model', df_all.shape[1])

	#One-hot-encoding features
	print('one-hot', datetime.now())
	ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'dac_month', 'dac_day', 'dac_weekday', 'dac_season', 'tfa_month', 'tfa_day', 'tfa_weekday', 'tfa_season']
if dataset_no == 1:
	#Loading data
	print('load data', datetime.now())
	df_train = pd.read_csv('Dataset_sql\\train_users_2_4.csv')

	#remove rows older than 1/1/2014
	#dac2 = df_train.date_account_created.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
	#df_train = df_train.drop(dac2[dac2 < datetime.strptime('20140101', '%Y%m%d')].index)

	#remove where !hassessions
	hassessions = df_train['HasSessions']
	df_train = df_train.drop(hassessions[hassessions == 0].index)
	print('rows', df_train.shape[0]) 

	df_test = pd.read_csv('Dataset_sql\\test_users_4.csv')
	labels = df_train['country_destination'].values
	df_train = df_train.drop(['country_destination'], axis=1)
	id_test = df_test['id']
	piv_train = df_train.shape[0]

	#Creating a DataFrame with train+test data
	df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
	
	#Removing id and date_first_booking
	df_all = df_all.drop(['id', 'date_first_booking', 'sessions_count', 'HasSessions'], axis=1)

	#Filling nan
	df_all = df_all.fillna(-1)

	#####Feature engineering#######
	print('features in the csv', df_all.shape[1])

	#date_account_created
	print('dac', datetime.now())
	dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
	df_all['dac_year'] = dac[:,0]
	df_all['dac_month'] = dac[:,1]
	df_all['dac_day'] = dac[:,2]
	dac2 = df_all.date_account_created.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
	df_all['dac_weekday'] = dac2.apply(lambda x: x.weekday())
	df_all['dac_season'] = dac2.apply(calculate_season)
	df_all = df_all.drop(['date_account_created'], axis=1)

	#timestamp_first_active
	print('tfa', datetime.now())
	tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
	df_all['tfa_year'] = tfa[:,0]
	df_all['tfa_month'] = tfa[:,1]
	df_all['tfa_day'] = tfa[:,2]
	df_all = df_all.drop(['timestamp_first_active'], axis=1)

	#Age
	print('age', datetime.now())
	av = df_all.age.values
	df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

	#remove features
	print('remove features', datetime.now())
	df_all = df_all.drop(['Sessions' + str(i) for i in [0]], axis=1)
	df_all = df_all.drop(['SessionsD' + str(i) for i in range(456)], axis=1)

	print('features in the model', df_all.shape[1])

	#One-hot-encoding features
	print('one-hot', datetime.now())
	ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'dac_season', 'sessions_preferred_device']
if dataset_no == 2:
	#Loading data
	print('load data', datetime.now())
	df_train = pd.read_csv('Dataset_sql\\train_users_2_4.csv')

	hassessions = df_train['HasSessions']
	df_train = df_train.drop(hassessions[hassessions == 0].index)
	print('rows', df_train.shape[0]) 

	df_test = pd.read_csv('Dataset_sql\\test_users_4.csv')
	labels = df_train['country_destination'].values
	df_train = df_train.drop(['country_destination'], axis=1)
	id_test = df_test['id']
	piv_train = df_train.shape[0]

	#Creating a DataFrame with train+test data
	df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
	#Removing id and date_first_booking
	df_all = df_all.drop(['id', 'date_first_booking', 'HasSessions'], axis=1)

	#Filling nan
	df_all = df_all.fillna(-1)

	#####Feature engineering#######
	print('features in the csv', df_all.shape[1])

	#date_account_created
	print('dac', datetime.now())
	dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
	df_all['dac_year'] = dac[:,0]
	df_all['dac_month'] = dac[:,1]
	df_all['dac_day'] = dac[:,2]

	#day of week, seazon
	print('dac2', datetime.now())
	dac2 = df_all.date_account_created.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
	df_all['dac_weekday'] = dac2.apply(lambda x: x.weekday())
	df_all['dac_season'] = dac2.apply(calculate_season)

	df_all = df_all.drop(['date_account_created'], axis=1)

	#timestamp_first_active
	print('tfa', datetime.now())
	tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
	df_all['tfa_year'] = tfa[:,0]
	df_all['tfa_month'] = tfa[:,1]
	df_all['tfa_day'] = tfa[:,2]
	df_all['tfa_hour'] = tfa[:,3]

	#day of week, season
	print('tfa2', datetime.now())
	tfa2 = df_all.timestamp_first_active.astype(str).apply(lambda x: datetime.strptime(x[:8], '%Y%m%d'))
	df_all['tfa_weekday'] = tfa2.apply(lambda x: x.weekday())
	df_all['tfa_season'] = tfa2.apply(calculate_season)

	df_all = df_all.drop(['timestamp_first_active'], axis=1)

	#Age
	print('age', datetime.now())
	av = df_all.age.values
	df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

	#remove features
	print('remove features', datetime.now())
	columns_removed = [0]
	df_all = df_all.drop(['Sessions' + str(i) for i in columns_removed], axis=1)
	df_all = df_all.drop(['SessionsD' + str(i) for i in range(456)], axis=1)

	print('features in the model', df_all.shape[1])

	#One-hot-encoding features
	print('one-hot', datetime.now())
	ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'dac_month', 'dac_day', 'dac_weekday', 'dac_season', 'tfa_month', 'tfa_day', 'tfa_weekday', 'tfa_season', 'sessions_preferred_device']

for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
y = labels

le = LabelEncoder()
le.fit(y)

Xy = []
for i in range(iterations_per_job):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	Xy.append([X_train, X_test, y_train, y_test])

def job_function(params):
	learning_rate = params[0]
	max_depth = params[1]
	ss_cs = params[2]
	gamma = params[3]
	min_child_weight = params[4]
	reg_lambda = params[5]
	reg_alpha = params[6]

	early_stopping_rounds = 25
	if learning_rate >= 0.3:
		early_stopping_rounds = 5
	if learning_rate <= 0.03:
		early_stopping_rounds = 50

	scores = []
	for i in range(iterations_per_job):
		X_train = Xy[i][0]
		X_test = Xy[i][1]
		y_train = Xy[i][2]
		y_test = Xy[i][3]
		
		y_train2 = le.transform(y_train)   
		y_test2 = le.transform(y_test)   

		clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='multi:softprob', subsample=ss_cs, colsample_bytree=ss_cs, gamma=gamma, min_child_weight=min_child_weight, seed=0, silent=True, reg_lambda=reg_lambda, reg_alpha=reg_alpha)      
		clf.fit(X_train, y_train, eval_set=[(X_test, y_test2)], eval_metric=calculate_score_2, early_stopping_rounds=early_stopping_rounds, verbose=False)
		y_predicted = clf.predict_proba(X_test, ntree_limit=clf.booster().best_ntree_limit)
		score = calculate_score(y_predicted, y_test2)
		scores.append(score)

	avg_score = np.array(scores).mean()
	print(avg_score, params)
	return avg_score

'''
values_learning_rate = [0.03]			#[0.01, 0.03, 0.1, 0.3] default 0.1
values_max_depth = [6,8]				    #[6, 8] default 6
values_ss_cs = [0.5]					#[0.5, 1] default 1
values_gamma = [0,1,2]					#default 0
values_min_child_weight = [1,2]			#default 1
values_lambda = [1, 2]					#default 1
values_alpha = [0, 1]					#default 0
'''
values_learning_rate = [0.03]			#[0.01, 0.03, 0.1, 0.3] default 0.1
values_max_depth = [6,8]				    #[6, 8] default 6
values_ss_cs = [0.5]					#[0.5, 1] default 1
values_gamma = [1]					#default 0
values_min_child_weight = [1,2]		#default 1
values_lambda = [1,2]					#default 1
values_alpha = [1]					#default 0

for learning_rate in values_learning_rate:
	for max_depth in values_max_depth:
		for ss_cs in values_ss_cs:
			for gamma in values_gamma:
				for min_child_weight in values_min_child_weight:
					for reg_lambda in values_lambda:
						for reg_alpha in values_alpha:
							job = [job_function, [learning_rate, max_depth, ss_cs, gamma, min_child_weight, reg_lambda, reg_alpha], 0]
							jobs.append(job)

print('Jobs', len(jobs))
threads = []
for i in range (thread_count):
	thread = myThread(i)
	thread.start()
	threads.append(thread)

for t in threads:
    t.join()

print('Jobs done')

'''
best_score = 0
for x in jobs:
	if x[2] > best_score:
		print('***Best Score***')
		best_score = x[2]
	print(x[2], x[1])
'''
'''
#learning_rate = 0.03
learning_rate = 0.3

early_stopping_rounds = 25
if learning_rate <= 0.03:
	early_stopping_rounds = 50

ss_cs = 0.5
#ss_cs = 1.0

#max_depth = 6
max_depth = 8

print(learning_rate, early_stopping_rounds, ss_cs, max_depth)

clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='multi:softprob', subsample=ss_cs, colsample_bytree=ss_cs, seed=0, silent=True)      
#default merror
#mlogloss
#clf.fit(X_train, y_train, eval_set=[(X_train, y_train2), (X_test, y_test2)], eval_metric='mlogloss', early_stopping_rounds=early_stopping_rounds)
#eval_metric=calculate_score_2
clf.fit(X_train, y_train, eval_set=[(X_train, y_train2), (X_test, y_test2)], eval_metric=calculate_score_2, early_stopping_rounds=early_stopping_rounds)
#ndcg - doesn't work without a Customized objective function in train.py
#clf.fit(X_train, y_train, eval_set=[(X_train, y_train2), (X_test, y_test2)], eval_metric='ndcg5', early_stopping_rounds=early_stopping_rounds)

#calculate score
#important: use the best tree
y_predicted = clf.predict_proba(X_test, ntree_limit=clf.booster().best_ntree_limit)
score = calculate_score(y_predicted, y_test2)
#score = round(100*score, 3)
print(score, 1-score)
'''




