import datetime
from datetime import datetime
import math
from math import sqrt
import threading

import numpy as np
import pandas as pd

from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

from xgboost.sklearn import XGBClassifier

def calculate_season(x):
	if x.month in (3,4,5): 
		return 0
	if x.month in (6,7,8):
		return 1
	if x.month in (9,10,11):
		return 2
	return 3

def calculate_top5(result):
	if(sum(result) < 0.99 or sum(result) > 1.01):
		raise ValueError ('what?')

	top5 = zip(np.argsort(result)[::-1][:5], np.sort(result)[::-1][:5])
	top5 = [x[0] for x in top5 if x[1] != 0] 
	
	if(len(top5) != 5):
		raise ValueError ('what?')

	return top5

thread_results = [0, 0]

def model1(df_train, df_test):
	print('model1')

	print('rows', df_train.shape[0]) 

	#remove rows with no sessions data
	hassessions = df_train['HasSessions']
	df_train = df_train.drop(hassessions[hassessions == 0].index)

	#remove rows older than 1/1/2014
	#dac2 = df_train.date_account_created.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
	#print('removing rows', len(dac2[dac2 < datetime.strptime('20140101', '%Y%m%d')].index))
	#df_train = df_train.drop(dac2[dac2 < datetime.strptime('20140101', '%Y%m%d')].index)

	print('rows', df_train.shape[0]) 

	labels = df_train['country_destination'].values
	df_train = df_train.drop(['country_destination'], axis=1)
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

	for f in ohe_feats:
		df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
		df_all = df_all.drop([f], axis=1)
		df_all = pd.concat((df_all, df_all_dummy), axis=1)

	#Splitting train and test
	vals = df_all.values
	X = vals[:piv_train]
	y = labels
	X_predict = vals[piv_train:]

	#learning_rate, max_depth, ss_cs, gamma, min_child_weight, reg_lambda, reg_alpha =  0.03, 6, 0.5, 2, 2, 2, 1
	learning_rate, max_depth, ss_cs, gamma, min_child_weight, reg_lambda, reg_alpha =  0.03, 8, 0.5, 2, 1, 2, 0

	early_stopping_rounds = 25
	if learning_rate <= 0.03:
		early_stopping_rounds = 50

	print(learning_rate, max_depth, ss_cs, gamma, min_child_weight, reg_lambda, reg_alpha)

	#n_estimators = 455
	n_estimators = 350
	#n_estimators = 1
	print(n_estimators)

	print('fit start', datetime.now())
	clf2 = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective='multi:softprob', subsample=ss_cs, colsample_bytree=ss_cs, gamma=gamma, min_child_weight=min_child_weight, seed=0, silent=True, reg_lambda=reg_lambda, reg_alpha=reg_alpha, nthread=-1)      
	clf2.fit(X, y)
	y_predicted2 = clf2.predict_proba(X_predict)  

	return y_predicted2

def model2(df_train, df_test):
	print('model2')

	print('rows', df_train.shape[0]) 

	labels = df_train['country_destination'].values
	df_train = df_train.drop(['country_destination'], axis=1)
	piv_train = df_train.shape[0]

	#Creating a DataFrame with train+test data
	df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

	#Removing id and date_first_booking
	df_all = df_all.drop(['id', 'date_first_booking', 'sessions_count', 'sessions_duration', 'sessions_devices', 'sessions_preferred_device', 'HasSessions'], axis=1)

	#remove features
	print('remove features', datetime.now())
	df_all = df_all.drop(['Sessions' + str(i) for i in range(456)], axis=1)
	df_all = df_all.drop(['SessionsD' + str(i) for i in range(456)], axis=1)

	#Filling nan
	df_all = df_all.fillna(-1)

	#####Feature engineering#######
	#date_account_created
	print('dac', datetime.now())
	dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
	df_all['dac_year'] = dac[:,0]
	df_all['dac_month'] = dac[:,1]
	df_all['dac_day'] = dac[:,2]
	#dac2 = df_all.date_account_created.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
	#df_all['dac_weekday'] = dac2.apply(lambda x: x.weekday())
	#df_all['dac_season'] = dac2.apply(calculate_season)
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

	print('features in the model', df_all.shape[1])

	#One-hot-encoding features
	print('one-hot', datetime.now())
	ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser'] 
	
	for f in ohe_feats:
		df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
		df_all = df_all.drop([f], axis=1)
		df_all = pd.concat((df_all, df_all_dummy), axis=1)

	#Splitting train and test
	vals = df_all.values
	X = vals[:piv_train]
	y = labels
	X_predict = vals[piv_train:]

	#learning_rate, max_depth, ss_cs, gamma, min_child_weight, reg_lambda, reg_alpha =  0.03, 6, 0.5, 2, 2, 2, 1
	learning_rate, max_depth, ss_cs, gamma, min_child_weight, reg_lambda, reg_alpha =  0.03, 8, 0.5, 2, 1, 2, 0
	
	early_stopping_rounds = 25
	if learning_rate <= 0.03:
		early_stopping_rounds = 50

	print(learning_rate, max_depth, ss_cs, gamma, min_child_weight, reg_lambda, reg_alpha)

	#n_estimators = 446
	n_estimators = 360
	#n_estimators = 1
	print(n_estimators)

	print('fit start', datetime.now())
	clf2 = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective='multi:softprob', subsample=ss_cs, colsample_bytree=ss_cs, gamma=gamma, min_child_weight=min_child_weight, seed=0, silent=True, reg_lambda=reg_lambda, reg_alpha=reg_alpha, nthread=-1)      
	clf2.fit(X, y)
	y_predicted2 = clf2.predict_proba(X_predict)  

	return y_predicted2

class myThread1 (threading.Thread):
	def __init__(self, df_train, df_test):
		threading.Thread.__init__(self)
		self.df_train = df_train
		self.df_test = df_test

	def run(self):
		np.random.seed(0)
		y_predicted = model1(self.df_train, self.df_test)
		thread_results[0] = y_predicted

class myThread2 (threading.Thread):
	def __init__(self, df_train, df_test):
		threading.Thread.__init__(self)
		self.df_train = df_train
		self.df_test = df_test

	def run(self):
		np.random.seed(0)
		y_predicted = model2(self.df_train, self.df_test)
		thread_results[1] = y_predicted

np.random.seed(0)

#Loading data
print('load data', datetime.now())
df_train = pd.read_csv('Dataset_sql\\train_users_2_4.csv')
df_test = pd.read_csv('Dataset_sql\\test_users_4.csv')

labels = df_train['country_destination'].values
le = LabelEncoder()
le.fit(labels)

id_test = df_test['id']
hassessions_test = df_test['HasSessions']

threads = []

thread1 = myThread1(df_train, df_test)
thread1.start()
threads.append(thread1)

thread2 = myThread2(df_train, df_test)
thread2.start()
threads.append(thread2)

for t in threads:
    t.join()

m1_y_predicted = thread_results[0]
m2_y_predicted = thread_results[1]

for w in range(21):
	#w1 = w*0.05
	w1 = 0.80 + w*0.01
	w2 = 1 - w1

	filename = 'results2m_a_' + str(w) + '.csv'

	ids = []  #list of ids
	cts = []  #list of countries
	for i in range(len(id_test)):
		idx = id_test[i]
		ids += [idx] * 5

		if(hassessions_test[i] == 0):
			y_predicted = m2_y_predicted[i]
		else:
			y_predicted = m1_y_predicted[i]*w1 + m2_y_predicted[i]*w2

		cts += le.inverse_transform(calculate_top5(y_predicted)).tolist()

	#Generate submission
	sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
	sub.to_csv(filename,index=False)

	filename = 'results2m_b_' + str(w) + '.csv'

	ids = []  #list of ids
	cts = []  #list of countries
	for i in range(len(id_test)):
		idx = id_test[i]
		ids += [idx] * 5

		y_predicted = m1_y_predicted[i]*w1 + m2_y_predicted[i]*w2

		cts += le.inverse_transform(calculate_top5(y_predicted)).tolist()

	#Generate submission
	sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
	sub.to_csv(filename,index=False)

