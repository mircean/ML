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

top_countries = ["NDF", "US", "other", "FR", "IT"]

thread_count = 10
train_and_test_scores = [x for x in range(thread_count)]

class myThread (threading.Thread):
	def __init__(self, threadID):
		threading.Thread.__init__(self)
		self.threadID = threadID
    
	def run(self):
		#print(self.threadID, 'Starting')
		myThreadFunc(self.threadID)
		#print(self.threadID, 'Done')

def calculate_top5(result):
	if(sum(result) < 0.99 or sum(result) > 1.01):
		raise ValueError ('what?')

	top5 = zip(np.argsort(result)[::-1][:5], np.sort(result)[::-1][:5])
	top5 = [x[0] for x in top5 if x[1] != 0] 
	
	i = 0
	while len(top5) < 5:
		if top_countries[i] not in top5:
			top5.append(top_countries[i])
		i += 1

	if(len(top5) != 5):
		raise ValueError ('what?')

	return top5

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

def calculate_season(x):
	if x.month in (3,4,5): 
		return 0
	if x.month in (6,7,8):
		return 1
	if x.month in (9,10,11):
		return 2
	return 3

np.random.seed(0)

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

#column_indexes = [-1] + [i for i in range(0,456)]
column_indexes = [-1] + [i for i in range(11,250)]
#1:50 - no
#250:456 - no

for i in column_indexes:
	#for train_test_split
	np.random.seed(0)
	if i in columns_removed:
		continue

	if i == -1:
		#print('compute baseline')
		score = 85.478
		print('baseline score', score)
		best_score = score
		best_index = -1
		continue
	else:
		column_name = 'Sessions' + str(i)
		print('processing column', column_name, datetime.now())
		df_all_2 = df_all
		df_all = df_all.drop([column_name], axis=1)

	#Splitting train and test
	vals = df_all.values
	X = vals[:piv_train]
	y = labels
	X_test = vals[piv_train:]

	le = LabelEncoder()
	le.fit(y)

	learning_rate, max_depth, ss_cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.3, 8, 0.5, 2, 2, 2, 1 
	early_stopping_rounds = 10

	Xy = []
	for j in range(thread_count):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		Xy.append([X_train, X_test, y_train, y_test])

	def myThreadFunc(ThreadID):
		X_train = Xy[ThreadID][0]
		X_test = Xy[ThreadID][1]
		y_train = Xy[ThreadID][2]
		y_test = Xy[ThreadID][3]
		
		y_train2 = le.transform(y_train)   
		y_test2 = le.transform(y_test)   

		clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='multi:softprob', subsample=ss_cs, colsample_bytree=ss_cs, gamma=gamma, min_child_weight=min_child_weight, seed=0, silent=True, reg_lambda=reg_lambda, reg_alpha=reg_alpha)      
		clf.fit(X_train, y_train, eval_set=[(X_test, y_test2)], eval_metric=calculate_score_2, early_stopping_rounds=early_stopping_rounds, verbose=False)
		y_predicted = clf.predict_proba(X_test, ntree_limit=clf.booster().best_ntree_limit)
		score = calculate_score(y_predicted, y_test2)
		#print(score, clf.booster().best_ntree_limit)
	
		train_and_test_scores[ThreadID] = score

	threads = []
	for j in range (thread_count):
		thread = myThread(j)
		thread.start()
		threads.append(thread)

	for t in threads:
		t.join()

	score = round(100*np.array(train_and_test_scores).mean(), 3)

	if(i == -1):
		print('baseline score', score)
		best_score = score
		best_index = -1
	else:
		print('mean score', score, 'delta', score - best_score)
		if score >= best_score + 0.01:
			print('*** Best Score ***', score)
			best_score = score
			best_index = i
		#restore df
		df_all = df_all_2

print('best index', best_index)








