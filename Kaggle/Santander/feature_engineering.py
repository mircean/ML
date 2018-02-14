from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import normalize

def feature_engineering(df_all, piv_train):
    #Removing id 
    df_all = df_all.drop(['ID'], axis=1)
    
    ### Feature engineering
    print('Feature engineering start')

    #new feature: no of zeros
    print('Adding count 0')
    df_all['n0'] = (df_all == 0).sum(axis=1)
    #print('features in the model', df_all.shape[1])

    #Filling nan
    print('Filling nan')
    #Replace nan with mean
    #the other option is to replace nan with -2 (value not used, better than 9999999999/-99999999 because it makes it easier to visualize the feature). mean gives higher score
    for x in df_all.columns:
        df_all[x] = df_all[x].replace(9999999999, np.NaN)
        df_all[x] = df_all[x].replace(-999999, np.NaN)

    imp = Imputer(strategy='mean', axis=0)
    df_all = pd.DataFrame(data=imp.fit_transform(df_all), columns=df_all.columns)

    print('Removing constant columns')
    df_train2 = df_all[:piv_train]
    columns_to_remove = []
    for x in df_train2.columns.values:
        if df_train2[x].std() == 0:
		    #print(x, 'is constant')
            columns_to_remove.append(x)

    df_all = df_all.drop(columns_to_remove, axis=1)
    print('features in the model', df_all.shape[1])

    print('Removing duplicate columns')
    df_train2 = df_all[:piv_train]
    columns_to_remove = []
    for i in range(df_all.shape[1]):
        column_i = df_all.columns.values[i]
        for j in range(i+1, df_all.shape[1]):
            column_j = df_all.columns.values[j]
            if np.array_equal(df_train2[column_i].values, df_train2[column_j].values):
                #print(column_i, 'equals', column_j)
                #columns_to_remove.append(column_i)
                columns_to_remove.append(column_j)
                break

    df_all = df_all.drop(columns_to_remove, axis=1)
    print('features in the model', df_all.shape[1])

    explore = 0
    if explore == 1:
        df_all_copy = df_all

	    #no nan
        for x in df_all.columns:
            if df_all[x].isnull().sum() > 0:
                print('nan found', x)

	    #
        #explore columns
	    #
	    #columns with min==0 and max<=999999 look OK
        print('features in the model', df_all.shape[1])
        columns_OK = []
        for x in df_all.columns:
            if df_all[x].min() == 0 and df_all[x].max() < 999999:
                columns_OK.append(x)

	    #244  columns min 0, max <= 999999
		    #178 columns min 0, max <= 1000
		    #56 columns min 0, max 1

        df_all = df_all.drop(columns_OK, axis=1)
        print('features in the model', df_all.shape[1])
	    #62 columns left

	    #columns with min, max not contain 999999 look OK
        columns_OK = []
        for x in df_all.columns:
            if str(df_all[x].min()).find('99999') == -1 and str(df_all[x].max()).find('99999') == -1:
                #df_all[x].describe()
                columns_OK.append(x)

        df_all = df_all.drop(columns_OK, axis=1)
        print('features in the model', df_all.shape[1])
	    #20 columns left

	    #9999999999, -999999 are np.nan
	    #replace with -2. cannot replace with -1 because -1 is used
	    #the other option is to set missing=9999999999 in clf; using -2 to make it easier to look at the feature (describe, value_counts, hist)
	    #or, replace -999999 with the most common value
        for x in df_all.columns:
            df_all[x] = df_all[x].replace(9999999999,-2)
            df_all[x] = df_all[x].replace(-999999,-2)
            df_all[x].describe()
            df_all[x].value_counts()	

	    #columns with only 0,-1 and -2 are likely poor features
	    #KBest only selects 2 of them in the worst 5% features
        print('Bad features')
        for x in df_all.columns:
            values = df_all[x].value_counts().keys().values
            values.sort()
            if np.array_equal(values, [-2, -1,  0]) or np.array_equal(values, [-2,  0]):
                print(x)
        '''
	    delta_imp_amort_var18_1y3
	    delta_imp_amort_var34_1y3
	    delta_num_reemb_var13_1y3
	    delta_num_reemb_var17_1y3
	    delta_num_reemb_var33_1y3
	    delta_num_trasp_var17_in_1y3
	    delta_num_trasp_var17_out_1y3
	    delta_num_trasp_var33_in_1y3
	    delta_num_trasp_var33_out_1y3
	    '''

	    #
	    #low variance, KBest
	    #
        df_all = df_all_copy
        for x in df_all.columns:
            df_all[x] = df_all[x].replace(9999999999,-2)
            df_all[x] = df_all[x].replace(-999999,-2)

        #features with low variance
        df_train2 = df_all[:piv_train]
        std = []
        for x in df_all.columns:
            std.append((df_train2[x].std(), x))

        std.sort()
        for x in range(20):
            print(std[x])
            #below 0.01. KBest doesn't select any of them in the worst 5% features

        '''
	    (0.005129183171326533, 'ind_var29')
	    (0.00512918317133366, 'ind_var13_medio')
	    (0.005129183171334088, 'ind_var18')
	    (0.005129183171334628, 'ind_var34')
	    (0.006281899464614709, 'ind_var7_emit_ult1')
	    '''

	    #for f_classif, not need to std
        X = df_all.vals[:piv_train]
        KBest = SelectPercentile(f_classif, percentile=95).fit(X, y)
        KBest_columns = KBest.get_support()

        print('Worst 5% features')
        columns_to_remove = [df_all.columns[x] for x in range(len(KBest_columns)) if KBest_columns[x] == False]
        for x in columns_to_remove:
            print(x)
        '''
	    imp_ent_var16_ult1
	    imp_op_var40_comer_ult3
	    imp_sal_var16_ult1
	    ind_var32
	    num_op_var40_ult3
	    num_var32
	    num_var37_med_ult2
	    saldo_var1
	    saldo_var32
	    delta_num_reemb_var33_1y3
	    delta_num_trasp_var33_out_1y3
	    num_var7_recib_ult1
	    num_trasp_var33_out_ult1
	    saldo_medio_var17_ult1
	    saldo_medio_var17_ult3
	    saldo_medio_var29_hace3
	    '''

        '''
        columns_to_remove = [x for x in range(len(KBest_columns)) if KBest_columns[x] == False]
        X = np.delete(X, columns_to_remove, axis=1)
        print('features in the model', X.shape[1])
        '''

        '''
        X_std = StandardScaler().fit_transform(X)
        X_bin = Binarizer().fit_transform(X_std)
        KBest2 = SelectPercentile(chi2, percentile=95).fit(X_bin, y)
        KBest2_columns = KBest2.get_support()

        lambda1=lambda x,y: x==True and y==True
        columns_to_keep = [lambda1(x[0],x[1]) for x in zip(KBest_columns, KBest2_columns)]
        or
        lambda1=lambda x,y: x==False and y==False
        columns_to_remove = [lambda1(x[0],x[1]) for x in zip(KBest_columns, KBest2_columns)]
        '''

    #Add PCA features
    print('Adding PCA features')
    pca = PCA(n_components=2)
    X = df_all.values
    #normalize together test+train data
    X_norm=normalize(X, axis=0)
    #PCA fit train data, tranform test data
    X_train=X_norm[:piv_train]
    X_test=X_norm[piv_train:]
    X_train_pca=pca.fit_transform(X_train)
    X_test_pca=pca.transform(X_test)
    df_all['PCA1']=np.append(X_train_pca[:,0], X_test_pca[:,0])
    df_all['PCA2']=np.append(X_train_pca[:,1], X_test_pca[:,1])
    print('features in the model', df_all.shape[1])

    '''
    #remove worst features
    print('Removing worst 5% features')
    X = df_all.values[:piv_train]
    y = labels
    KBest = SelectPercentile(f_classif, percentile=98).fit(X, y)
    KBest_columns = KBest.get_support()

    columns_to_remove = [df_all.columns[x] for x in range(len(KBest_columns)) if KBest_columns[x] == False]
    df_all = df_all.drop(columns_to_remove, axis=1)
    print('features in the model', df_all.shape[1])
    '''

    #var38
    print('Processing var38')
    df_all['var38_1'] = [1 if round(x, 6) == 117310.979016 else 0 for x in df_all['var38']]
    df_all['var38_2'] = [np.log(x) if round(x, 6) != 117310.979016 else 0 for x in df_all['var38']]
    df_all = df_all.drop('var38', axis=1)
    print('features in the model', df_all.shape[1])

    '''
    #age buckets
    print('Processing var15')
    #for i in [x for x in range(0,5)]:
    #	df_all['age_' + str(i)] = df_all.age.apply(lambda x: 1 if 20*i < x <= 20*(i + 1) else 0)
    df_all['age_under20'] = df_all['var15'].apply(lambda x: 1 if x < 20 else 0)
    df_all['age_2030'] = df_all['var15'].apply(lambda x: 1 if 20 <= x < 30 else 0)
    df_all['age_3040'] = df_all['var15'].apply(lambda x: 1 if 30 <= x < 40 else 0)
    df_all['age_4050'] = df_all['var15'].apply(lambda x: 1 if 40 <= x < 50 else 0)
    df_all['age_5060'] = df_all['var15'].apply(lambda x: 1 if 50 <= x < 60 else 0)
    df_all['age_over60'] = df_all['var15'].apply(lambda x: 1 if x >= 60 else 0)
    print('features in the model', df_all.shape[1])
    '''

    '''
    #saldo_var30
    print('Processing saldo_var30')
    df_all['saldo_var30_log'] = df_all['saldo_var30'].apply(lambda x: np.log(x) if x > 0 else 0)
    print('features in the model', df_all.shape[1])
    '''

    '''
    print('Processing saldo_var5*')
    df_all['saldo_medio_var5_hace_calc'] = df_all['saldo_medio_var5_hace3'] + df_all['saldo_medio_var5_hace2']
    df_all['saldo_medio_var5_ult_calc1'] = df_all['saldo_medio_var5_ult3'] + df_all['saldo_medio_var5_ult1']
    df_all['saldo_medio_var5_ult_calc2'] = df_all['saldo_medio_var5_ult3'] - df_all['saldo_medio_var5_ult1']
    print('features in the model', df_all.shape[1])
    '''
    '''
    #append stacking
    print('Append stacking')
    df_train2 = pd.read_csv('train2.csv')
    df_test2 = pd.read_csv('test2.csv')
    df_all2 = pd.concat((df_train2, df_test2), axis=0, ignore_index=True)
    df_all = pd.concat((df_all, df_all2), axis=1)
    print('features in the model', df_all.shape[1])
    '''

    print('Feature engineering done')
    ### Feature engineering done

    return df_all

