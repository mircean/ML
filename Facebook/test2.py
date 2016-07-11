import sys
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

np.random.seed(0)

print('load data', datetime.now())
df_train = pd.read_csv('Dataset\\train.csv', index_col = 0)

'''
i = 0
folds = KFold(df_train.shape[0], n_folds=5, shuffle=True)
for train, test in folds:
    df_train_fold = df_train.iloc[train]
    df_test_fold = df_train.iloc[test]
    df_train_fold.to_csv('train' + str(i) + '.csv', index = True)
    df_test_fold.to_csv('test' + str(i) + '.csv', index = True)
    i += 1
    #break
'''
train_len = int(df_train.shape[0]*0.8)
df_train_sorted = df_train.sort_values('time')
df_train_fold = df_train_sorted[:train_len].sort_index()
df_test_fold = df_train_sorted[train_len:].sort_index()
df_train_fold.to_csv('train1.csv', index = True)
df_test_fold.to_csv('test1.csv', index = True)
