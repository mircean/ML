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

def calculate_language(lang1, lang2, isUS):
	if lang1 == lang2:
		return 0
	if lang1 != 'en':
		if lang2 == 'en':
			if isUS == 1:
				return 0
			else:
				return 20
		else:
			return 100
	else: #lang1 == 'en':
		if lang2 == 'de':
			return 73
		if lang2 == 'es':
			return 92
		if lang2 == 'fr':
			return 92
		if lang2 == 'it':
			return 89
		if lang2 == 'nl':
			return 63
		if lang2 == 'pt':
			return 95

	raise ValueError ('what?')
	return 100

np.random.seed(0)

#dataset_no = 0 #w/o sessions, latest features
dataset_no = 1 #sessions, drop !hassessions, no latest features
#dataset_no = 2 #sessions, drop !hassessions, latest features

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
#df_all['first_affiliate_tracked'] = df_all['first_affiliate_tracked'].fillna('untracked')
df_all = df_all.fillna(-1)

#####Feature engineering#######
print('features in the csv', df_all.shape[1])

if dataset_no == 1:
	df_all = df_all.drop(['sessions_count'], axis=1)

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
	columns_removed = [0]
	df_all = df_all.drop(['Sessions' + str(i) for i in columns_removed], axis=1)
	df_all = df_all.drop(['SessionsD' + str(i) for i in range(456)], axis=1)

	print('features in the model', df_all.shape[1])

	#One-hot-encoding features
	print('one-hot', datetime.now())
	ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'dac_season', 'sessions_preferred_device']
else:
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
	#df_all['age'] = df_all.age.apply(lambda x: 2015 - x if x > 1900 else x)
	av = df_all.age.values
	df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

	#age buckets
	#for i in range(2,20):
	#	df_all['age_' + str(i)] = df_all.age.apply(lambda x: 1 if 5*i < x <= 5*(i + 1) else 0)

	#replace unknown with average
	#df_all['age_correct'] = df_all.age.apply(lambda x: 1 if x != -1 else 0)
	#age_avg = round(np.array([x for x in df_all.age if x != -1]).mean())
	#df_all['age'] = df_all.age.apply(lambda x: age_avg if x == -1 else x)

	#age2
	#df_all['age2'] = df_all.age.apply(lambda x: x*x if x != -1 else x)
	#df_all['age2'] = df_all.age.apply(lambda x: round(sqrt(x)) if x != -1 else x)

	#Language
	print('languages', datetime.now())
	'''
	df_all['language_us'] = df_all.language.apply(calculate_language, args = ('en', 1,))
	df_all['language_au'] = df_all.language.apply(calculate_language, args = ('en', 0,))
	df_all['language_ca'] = df_all.language.apply(calculate_language, args = ('en', 0,))
	df_all['language_gb'] = df_all.language.apply(calculate_language, args = ('en', 0,))

	df_all['language_de'] = df_all.language.apply(calculate_language, args = ('de', 0,))
	df_all['language_es'] = df_all.language.apply(calculate_language, args = ('es', 0,))
	df_all['language_fr'] = df_all.language.apply(calculate_language, args = ('fr', 0,))
	df_all['language_it'] = df_all.language.apply(calculate_language, args = ('it', 0,))
	df_all['language_nl'] = df_all.language.apply(calculate_language, args = ('nl', 0,))
	df_all['language_pt'] = df_all.language.apply(calculate_language, args = ('pt', 0,))
	'''

	#df_all['language_asian'] = df_all.language.apply(lambda x: 1 if x in ('zh', 'jp', 'ko', 'th', 'id') else 0)
	#df_all['language_n_eu'] = df_all.language.apply(lambda x: 1 if x in ('de', 'sv', 'nl', 'da', 'pl', 'cs', 'no', 'fi', 'is') else 0)

	#is_apple_user
	#print('is_apple_user', datetime.now())
	#df_all['is_apple_user'] = df_all.first_device_type.apply(lambda x: 1 if x in ('Mac Desktop', 'iPhone', 'iPad') else 0)

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
X_test = vals[piv_train:]

le = LabelEncoder()
le.fit(y)

learning_rate, max_depth, ss_cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.03, 8, 0.5, 2, 2, 2, 1 
early_stopping_rounds = 50

print(learning_rate, max_depth, ss_cs, gamma, min_child_weight, reg_lambda, reg_alpha)

Xy = []
for i in range(thread_count):
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
	print(score, clf.booster().best_ntree_limit)
	
	train_and_test_scores[ThreadID] = score

threads = []
for i in range (thread_count):
	thread = myThread(i)
	thread.start()
	threads.append(thread)

for t in threads:
    t.join()

print(train_and_test_scores)
print('mean score', round(100*np.array(train_and_test_scores).mean(), 3))
for x in train_and_test_scores:
	print(x)


'''
#Classifier
print('fit start', datetime.now())
clf = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0, nthread=8) 
clf.fit(X, y)
print('fit done', datetime.now())

y_pred = clf.predict_proba(X_test)  
#Taking the 5 classes with highest probabilities
y_pred_top5 = calculate_top5(y_pred, le)

ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += y_pred_top5[i]
    #cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('C:\\Users\mircean\OneDrive\Airbnb\\results.csv',index=False)
'''






