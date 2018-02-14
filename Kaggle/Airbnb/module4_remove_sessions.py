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


'''
use random_state in test_train_split so can compare numbers 
test size 30%
try nthread=16 when iterations = 1?
'''

top_countries = ["NDF", "US", "other", "FR", "IT"]

iterations = 3
train_and_test_scores = [x for x in range(iterations)]

class myThread (threading.Thread):
    def __init__(self, threadID, X, y):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.X = X
        self.y = y
    
    def run(self):
        #print(self.threadID, 'Starting')
        clf = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  
        score = train_and_test(clf, self.X, self.y, self.threadID)
        train_and_test_scores[self.threadID] = score
        #print(self.threadID, 'Done')

def calculate_season(x):
	if x.month in (3,4,5): 
		return 0
	if x.month in (6,7,8):
		return 1
	if x.month in (9,10,11):
		return 2
	return 3

def calculate_top5(y ,le):
	y_top5 = []

	for result in y:
		if(sum(result) < 0.99 or sum(result) > 1.01):
			raise ValueError ('what?')

		top5 = zip(le.inverse_transform(np.argsort(result)[::-1][:5]), np.sort(result)[::-1][:5])
		top5 = [x[0] for x in top5 if x[1] != 0] 
	
		i = 0
		while len(top5) < 5:
			if top_countries[i] not in top5:
				top5.append(top_countries[i])
			i += 1

		if(len(top5) != 5):
			raise ValueError ('what?')

		y_top5.append(top5)

	return y_top5

def calculate_score(y, y_top5):
	score_per_user = []
	for i in range(len(y)):
		actual_result = le.inverse_transform(y[i])

		user_score = 0
		for j in range(5):
			if(y_top5[i][j] == actual_result):
				user_score = user_score + 1/math.log2(j+2)
				break

		score_per_user.append(user_score)

	score = np.array(score_per_user).mean()
	return score

def train_and_test(clf, X, y, threadID):
	#print('train_and_test start', datetime.now())

	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=threadID)

	clf.fit(X_train, y_train)

	#calculate percent of accurate results
	#y_result = clf.predict(X_test)
	#y_accurate = [1 if x[0] == x[1] else 0 for x in zip(y, y_result)]
	#score = np.array(y_accurate).mean()
	#print(score)

	#calculate score
	y_result2 = clf.predict_proba(X_test)
	y_top5 = calculate_top5(y_result2, le)
	score = calculate_score(y_test, y_top5)
	score = round(100*score, 3)
	#print(score)

	#print('train_and_test done', datetime.now())
	return score

np.random.seed(0)

#Loading data
print('load data', datetime.now())
df_train = pd.read_csv('C:\\Users\mircean\OneDrive\Airbnb\Dataset\\train_users_2_sql.csv')

#remove rows older than 1/1/2014
dac2 = df_train.date_account_created.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
print('removing rows', len(dac2[dac2 < datetime.strptime('20140101', '%Y%m%d')].index))
df_train = df_train.drop(dac2[dac2 < datetime.strptime('20140101', '%Y%m%d')].index)
print('remaining rows', df_train.shape[0]) 

df_test = pd.read_csv('C:\\Users\mircean\OneDrive\Airbnb\Dataset\\test_users_sql.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]

#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking', 'sessions_count'], axis=1)

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

print('features in the model', df_all.shape[1])

#One-hot-encoding features
print('one-hot', datetime.now())
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'dac_season', 'sessions_preferred_device'] #'tfa_season'

for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

#remove features
features_removed = [0]
print('remove features', datetime.now())
df_all = df_all.drop(['Sessions' + str(i) for i in features_removed], axis=1)

features_to_remove = []

#column_indexes = [-1] + [i for i in range(0,456)]
column_indexes = [-1] + [i for i in range(319,456)]
for i in column_indexes:
	if i in features_removed:
		print('feature', i, 'already removed')
		continue
	
	if i == -1:
		print('compute baseline')
	else:
		column_name = 'Sessions' + str(i)
		print('processing column', column_name, datetime.now())
		df_all_2 = df_all
		df_all = df_all.drop([column_name], axis=1)

	#Splitting train and test
	vals = df_all.values
	X = vals[:piv_train]
	le = LabelEncoder()
	y = le.fit_transform(labels)   
	X_test = vals[piv_train:]

	#train, test
	threads = []
	for i1 in range (iterations):
		thread = myThread(i1, X, y)
		thread.start()
		threads.append(thread)

	for t in threads:
		t.join()

	score = round(np.array(train_and_test_scores).mean(), 3)

	if(i == -1):
		print('baseline score', score)
		best_score = score
	else:
		print('mean score', score, 'delta', score - best_score)
		if score >= best_score + 0.01:
			print('*** Best Score ***', score)
			best_score = score
			features_to_remove.append(column_name)
		else:
			#restore df
			df_all = df_all_2

print(features_to_remove)
