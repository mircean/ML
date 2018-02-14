import math
import datetime
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.multiclass import OneVsRestClassifier

from xgboost.sklearn import XGBClassifier

from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

countries_list = ["NDF", "US", "other", "FR", "IT", "AU", "CA", "DE", "ES", "GB", "NL", "PT"]
countries_dict = { x[0]:x[1] for x in zip(countries_list, range(len(countries_list))) }
top_countries = ["NDF", "US", "other", "FR", "IT"]

#id,date_account_created,timestamp_first_active,date_first_booking,gender,age,signup_method,signup_flow,language,affiliate_channel,affiliate_provider,first_affiliate_tracked,signup_app,first_device_type,first_browser,
#country_destination

features_to_remove = {'id', 'date_account_created', 'timestamp_first_active', 'date_first_booking', 'days_active_before_account_created', 'estimated_month_first_booking', 'sessions_count'}
features_categorical = {'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'sessions_preferred_device'}

def read_train_data():
	print('read_data', datetime.datetime.now())
	with open('C:\\Users\mircean\OneDrive\Airbnb\Dataset\\train_users_2.csv', 'r') as f:
#	with open('C:\\Users\mircean\OneDrive\Airbnb\\sql_train_users_2.csv', 'r') as f:
		data_iter = csv.reader(f, delimiter = ',')
		data = [data for data in data_iter]

	data_array = np.asarray(data)
	return data_array

def read_contest_data():
	print('read_data', datetime.datetime.now())
#	with open('C:\\Users\mircean\OneDrive\Airbnb\\sql_test_users.csv', 'r') as f:
	with open('C:\\Users\mircean\OneDrive\Airbnb\Dataset\\test_users.csv', 'r') as f:
		data_iter = csv.reader(f, delimiter = ',')
		data = [data for data in data_iter]

	data_array = np.asarray(data)
	return data_array

def write_results(users, y):
	print('write_results', datetime.datetime.now())
	with open('C:\\Users\mircean\OneDrive\Airbnb\\results.csv', 'w') as f_out:
		f_out.write("id,country" + "\n")
		for i in range(len(users)):
			for x in y[i]:
				f_out.write(users[i] + "," + x + "\n")

def prepare_numeric_data(features, data, plot = 0):
	print('prepare_numeric_data', datetime.datetime.now())
	#X_numeric contains numeric features. matrix with rows = no of samples, columns = <numeric features>
	X_numeric = np.empty([len(data), 0], dtype = float)

	iSessions = -1000
	for i in range(len(features)):
		new_column_name = features[i]
		if(new_column_name == 'Sessions0'):
			iSessions = i
			break

	for i in range(len(features)):
		if(not (iSessions <= i < iSessions + 456) and features[i] not in features_to_remove and features[i] not in features_categorical and features[i] != 'country_destination'):
			new_column_name = features[i]
			new_column = np.array([0 if x == '' else float(x) for x in data[:, i]])

			if(plot):
				df = pd.DataFrame(new_column, columns=[new_column_name])
				df.plot(kind='hist')
				plt.show()

			if(new_column_name == 'age'):
				#age > 1900: likely mistake, actually year of birth not  age
				new_column = np.array([2015 - x if x > 1900 else x for x in new_column])
				#set age to 0 if age < 12 or age > 120
				new_column = np.array([0 if x <= 12 or x > 120 else x for x in new_column])
				#replace 0 with mean
				#todo: mean in the train set, or train+test? latter seems better
				#imp = Imputer(missing_values=0, strategy='mean', axis=1) #axis=1 -> row
				#new_column = imp.fit_transform([new_column])[0]
				#new_column = np.array([round(x) for x in new_column])

				if(plot):
					df = pd.DataFrame(new_column, columns=[new_column_name])
					df.plot(kind='hist')
					plt.show()

			X_numeric = np.c_[X_numeric, new_column]

	if(iSessions != -1000):
		X_numeric = np.c_[X_numeric, data[:, iSessions : iSessions + 456]]

	return X_numeric

def prepare_categorical_data(features, data, features_encoding, plot = 0):
	print('prepare_categorical_data', datetime.datetime.now())
	#X_categorical contains categorical features. starts with matrix with rows = no of samples, columns = 0
	X_categorical = np.empty([len(data), 0], dtype = float)

	isTrain = 0
	if(len(features_encoding) == 0):
		isTrain = 1

	for i in range(len(features)):
		if(features[i] in features_categorical):
			new_column_name = features[i]
			new_column = data[:, i]

			if(plot):
				df = pd.DataFrame(new_column, columns=[new_column_name])
				df.groupby(new_column).size().plot(kind='bar')
				plt.show()

				if(new_column_name == 'language'):
					new_column_test = [x for x in new_column if x != 'en']

					df = pd.DataFrame(new_column_test, columns=[new_column_name])
					df.groupby(new_column_name).size().plot(kind='bar')
					plt.show()

			if(isTrain):
				dictionary = {}
				iCategory = 0
				for x in new_column:
					if(x not in dictionary):
						dictionary[x] = iCategory
						iCategory = iCategory + 1

				features_encoding[new_column_name] = dictionary
			else:
				for x in new_column:
					if(x not in features_encoding[new_column_name]):
						print(new_column_name, x)

				#todo: might be better to ignore
				if(new_column_name == 'signup_method'):
					new_column = np.array(['basic' if x == 'weibo' else x for x in new_column])

				if(new_column_name == 'signup_flow'):
					new_column = np.array(['0' if x == '14' else x for x in new_column])

				if(new_column_name == 'language'):
					new_column = np.array(['en' if x == '-unknown-' else x for x in new_column])

				if(new_column_name == 'first_browser'):
					new_column = np.array(['-unknown-' if x not in features_encoding[new_column_name] else x for x in new_column])

			new_column = np.array([features_encoding[new_column_name][x] for x in new_column])
			X_categorical = np.c_[X_categorical, new_column]
		
	return X_categorical, features_encoding

def prepare_train_data():
	print('prepare_train_data', datetime.datetime.now())
	data = read_train_data()
	features = data[0]
	data = data[1:]
	#for now, use the top 1000 rows
	#data = data[0:200]

	X_numeric = prepare_numeric_data(features, data)
	X_categorical, features_encoding = prepare_categorical_data(features, data, {})

	enc_X = preprocessing.OneHotEncoder(sparse=False)
	X_categorical_2 = enc_X.fit_transform(X_categorical)  
	X = np.c_[X_numeric, X_categorical_2]

	le = LabelEncoder()
	y = data[:, 15]
	y = le.fit_transform(y)

	return X, y, features_encoding, enc_X, le

def prepare_test_data(features_encoding, enc_X):
	print('prepare_test_data', datetime.datetime.now())
	data = read_contest_data()
	features = data[0]
	data = data[1:]

	users = data[:, 0]

	X_numeric = prepare_numeric_data(features, data)
	X_categorical, features_encoding = prepare_categorical_data(features, data, features_encoding)

	X_categorical_2 = enc_X.transform(X_categorical)  
	X_predict = np.c_[X_numeric, X_categorical_2]

	return X_predict, users

def calculate_top5(y ,le):
	y_top5 = []

	for result in y:
		if(sum(result) < 0.99 or sum(result) > 1.01):
			raise ValueError ('what?')

		#k = len([x for x in result if x > 0])
		#if(k < 5):
		#	1+1

		top5 = zip(le.inverse_transform(np.argsort(result)[::-1][:5]), np.sort(result)[::-1][:5])
		top5 = [x[0] for x in top5 if x[1] != 0] 
	
		iTopCountries = 0
		while len(top5) < 5:
			if top_countries[iTopCountries] not in top5:
				top5.append(top_countries[iTopCountries])
			iTopCountries = iTopCountries + 1

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

def my_scoring(clf, X_test, y_test):
	print('my_scoring test size', len(y_test))

	y_result = clf.predict(X_test)
	y_accurate = [1 if x[0] == x[1] else 0 for x in zip(y_test, y_result)]
	score = np.array(y_accurate).mean()
	print(score)

	y_result2 = clf.predict_proba(X_test)
	y_top5 = calculate_top5(y_result2, le)
	score = calculate_score(y_test, y_top5)
	print(score)

	return score

def train_and_test(clf, X, y):
	print('train_and_test', datetime.datetime.now())

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	clf.fit(X_train, y_train)
	y_result = clf.predict(X_test)

	#calculate percent of accurate results
	y_accurate = [1 if x[0] == x[1] else 0 for x in zip(y, y_result)]
	score = np.array(y_accurate).mean()
	print(score)

	y_result2 = clf.predict_proba(X_test)
	y_top5 = calculate_top5(y_result2, le)
	score = calculate_score(y_test, y_top5)
	print(score)

def train_compare_classifiers(X, y):
	print('train_compare_classifier', datetime.datetime.now())
	names = [
	#	"Nearest Neighbors", 
	#	"Linear SVM", 
	#	"RBF SVM", 
#		"Decision Tree",
		"Random Forest", 
	#	"OneVsAll Linear SVM"
	#	"AdaBoost", 
	#	"Naive Bayes", 
	#	"Linear Discriminant Analysis",
	#    "Quadratic Discriminant Analysis"
		]
	classifiers = [
	#   KNeighborsClassifier(3),
	#	SVC(kernel="linear", C=0.025),
	#   SVC(gamma=2, C=1),
#		DecisionTreeClassifier(max_depth=5),
		RandomForestClassifier(max_depth=5, max_features=1, n_jobs=-1),
	#	OneVsRestClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
	#   AdaBoostClassifier(),
	#   GaussianNB(),
	#   LinearDiscriminantAnalysis(),
	#   QuadraticDiscriminantAnalysis()
		]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	for name, clf in zip(names, classifiers):
		print(name, datetime.datetime.now())

		clf.fit(X_train, y_train)
		y_result = clf.predict(X_test)

		#calculate percent of accurate results
		y_accurate = [1 if np.array_equal(x[0], x[1]) else 0 for x in zip(y_test, y_result)]
		score = np.array(y_accurate).mean()
		print(score)

		y_result2 = clf.predict_proba(X_test)
		y_top5 = calculate_top5(y_result2)

		score = calculate_score(y_test, y_top5)
		print(score)

def train_cross_validate(clf, X, y):
	print('Cross validation', datetime.datetime.now())
	scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring=my_scoring)
	print('Mean', scores.mean())
	print('Cross validation done', datetime.datetime.now())

def main():
	print('Start', datetime.datetime.now())

	#train data
	X, y, features_encoding, enc_X, le = prepare_train_data()
	#test data
	#X_test, users = prepare_test_data(features_encoding, enc_X)

	#clf = DecisionTreeClassifier(max_depth=5)
	#train_and_test(clf, X, y)
	#clf = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0) 
	#train_and_test(clf, X, y)

	clf = DecisionTreeClassifier(max_depth=5)
	train_cross_validate(clf, X, y)
	clf = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0) 
	train_cross_validate(clf, X, y)

	1+1
	#train_compare_classifiers(X, y)

	#write results
	#clf = DecisionTreeClassifier(max_depth=5)
	clf = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0) 

	print('fit', datetime.datetime.now())
	clf.fit(X, y)

	print('predict', datetime.datetime.now())
	y_test = clf.predict_proba(X_test)
	y_top5 = calculate_top5(y_test, le)

	write_results(users, y_top5)

print('Module2')
#main()

'''
data = read_contest_data()
features = data[0]
data = data[1:]
X_categorical, features_encoding = prepare_categorical_data(features, data, {}, 1)
'''