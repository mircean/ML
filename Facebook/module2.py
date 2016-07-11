import sys
import datetime
import math
import numpy as np
import pandas as pd
import queue 
import json

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from xgboost.sklearn import XGBClassifier
from xgboost.core import DMatrix
from xgboost.training import train, cv

from sklearn.ensemble import RandomForestClassifier

from common import merge2, run_tasks

###consts
eps = 0.00001  #required to avoid some divisions by zero.

###parameters
split_by_time = 1
full_dataset = 1
n_topx = 7 #used in get_top5, was 5
n_threads = 10

test_max_cells = 500000
skip_in_cv = 11
cv_start_at = 5

### for cells/size
n_cells = 80
x_size = 10/n_cells
y_size = 10/n_cells

n_places = 0
n_places_th = 5
n_places_percentage = 0.9

### for clf
train_test_cv = 1
train_test = 1
n_folds = 5

classifier = 'xgb'

xgb_calculate_n_estimators = False
n_estimators_fixed = 100
learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.1, 6, 0.5, 0.5, 0, 1, 1, 0
early_stopping_rounds = 20

rf_n_estimators = 500 #1000
###done parameters

one_cell = 0
few_cells = 0
grid_search = 0
do_cv = 0
do_score = 0
x_min, x_max = 0, 10

def fe_time_1(df):
    #df['min']=df.time%60
    df['hour']=df.time//60%24 + (df.time%60)/60
    df['day']=df.time//(60*24)
    df['weekday'] = df.time//(60*24)%7
    df['month'] = df['time']//(60*24*30)%12 #rough estimate, month = 30 days
    df['year'] = df['time']//(60*24*30*12)

    df = df.drop(['time'], axis=1)
 
    return df

def fe_2(df):
    df['hour']=df.time//60%24 + (df.time%60)/60
    #df['day']=df.time//(60*24)
    df['weekday'] = df.time//(60*24)%7
    df['month'] = df['time']//(60*24*30)%12 #rough estimate, month = 30 days
    df['year'] = df['time']//(60*24*365)

    df = df.drop(['time'], axis=1)
 
    return df

def get_top5(le, y_predicted):
    dict = {}
    for place, score in zip(le.inverse_transform(np.argsort(y_predicted)[::-1][:n_topx]), np.sort(y_predicted)[::-1][:n_topx]):
        dict[str(place)] = str(score)
   
    return json.dumps(dict)

def calculate_score_per_row(y_predicted, y_true):
    top3 = np.argsort(y_predicted)[::-1][:3]
    score = 0
    for i in range(3):
        if(top3[i] == y_true):
            score = 1/(i + 1)
            break
    return score

def calculate_score_per_row2(y_predicted, y_true):
    score = 0
    for i in range(3):
        if(y_predicted[i] == y_true):
            score = 1/(i + 1)
            break
    return score

def calculate_score(y_predicted, y_true):
    score = calculate_score2(y_predicted, y_true.get_label())
    score = 1 - score
    return 'myscore', score

def calculate_score2(y_predicted, y_true):
    scores = [calculate_score_per_row(x[0], x[1]) for x in zip(y_predicted, y_true)]
    score = np.array(scores).mean()
    return score

#Keep in sync with m1
def get_cell(df_train, df_test, x1, y1):
    x2, y2 = round(x1 + x_size, 2), round(y1 + y_size, 2)

    df_train_cell = df_train[(df_train.x >= x1) & (df_train.x < x2) & (df_train.y >= y1) & (df_train.y < y2)]
    df_test_cell = df_test[(df_test.x >= x1) & (df_test.x < x2) & (df_test.y >= y1) & (df_test.y < y2)]

    return df_train_cell, df_test_cell

#Keep in sync with m1
def do_cell(task):
    df_train, df_test, x_start, y_start = task[0], task[1], task[2], task[3]
    n_places_th_local = n_places_th
    n_places_local = n_places

    if n_places != 0:
        tmp = df_train.shape[0]
        value_counts = df_train.place_id.value_counts()[0:n_places]
        df_train = pd.merge(df_train, pd.DataFrame(value_counts), left_on='place_id', right_index=True)[df_train.columns]
        n_places_th_local = value_counts.values[n_places - 1]
        percentage = df_train.shape[0]/tmp

    elif n_places_th != 0:
        value_counts = df_train.place_id.value_counts()
        n_places_local = value_counts[value_counts >= n_places_th_local].count()
        mask = value_counts[df_train.place_id.values] >= n_places_th_local
        percentage = mask.value_counts()[True]/df_train.shape[0]
        df_train = df_train.loc[mask.values]

    else:
        n_places_th_local = 2

        value_counts = df_train.place_id.value_counts()
        n_places_local = value_counts[value_counts >= n_places_th_local].count()
        mask = value_counts[df_train.place_id.values] >= n_places_th_local
        percentage = mask.value_counts()[True]/df_train.shape[0]

        while percentage > n_places_percentage:
            n_places_th_local += 1
            n_places_local = value_counts[value_counts >= n_places_th_local].count()
            mask = value_counts[df_train.place_id.values] >= n_places_th_local
            percentage = mask.value_counts()[True]/df_train.shape[0]

        n_places_th_local -= 1
        n_places_local = value_counts[value_counts >= n_places_th_local].count()
        mask = value_counts[df_train.place_id.values] >= n_places_th_local
        percentage = mask.value_counts()[True]/df_train.shape[0]

        df_train = df_train.loc[mask.values]


    #print(x_start, y_start, n_places_local, n_places_th_local, percentage)        

    #test
    row_ids = df_test.index
    if 'place_id' in df_test.columns:
        df_test = df_test.drop(['place_id'], axis=1)

    le = LabelEncoder()
    y = le.fit_transform(df_train.place_id.values)
    X = df_train.drop(['place_id'], axis=1).values
    X_predict = df_test.values

    score = 0
    n_estimators = 0
    if classifier == 'xgb':   
        if xgb_calculate_n_estimators == True:
            clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='multi:softprob', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)

            if train_test == 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
                clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=calculate_score, early_stopping_rounds=early_stopping_rounds, verbose=100 if one_cell == 1 else False)
                score = round(1 - clf.booster().best_score, 6)
                n_estimators = clf.booster().best_ntree_limit
            else:
                abc += 1
                xgb_options = clf.get_xgb_params()
                xgb_options['num_class'] = n_places + 1
                train_dmatrix = DMatrix(X, label=y)

                #some of the classes have less than n_folds, cannot use stratified KFold
                #folds = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
                folds = KFold(len(y), n_folds=n_folds, shuffle=True)
                cv_results = cv(xgb_options, train_dmatrix, clf.n_estimators, early_stopping_rounds=early_stopping_rounds, verbose_eval=100 if one_cell == 1 else False, show_stdv=False, folds=folds, feval=calculate_score)

                n_estimators = cv_results.shape[0]
                score = round(1 - cv_results.values[-1][0], 6)
                std = round(cv_results.values[-1][1], 6)
        else:
            n_estimators = n_estimators_fixed

        clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective='multi:softprob', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)
    elif classifier == 'rf':
        clf = RandomForestClassifier(n_estimators = rf_n_estimators, n_jobs = -1)
    else:
        abc += 1
    
    #if few_cells == 1 or grid_search == 1:
    #    return [score, None, None]

    clf.fit(X, y)
    y_predict = clf.predict_proba(X_predict)

    checkins = []
    for x in y_predict:
        checkins.append(get_top5(le, x))
        
    print(x_start, y_start, score, n_estimators, n_places_local, n_places_th_local, percentage)

    return [score, row_ids, checkins]

def do_cell_cv(task):
    df_train, df_test, df_train_cell, df_test_cell, x1, y1 = task[0], task[1], task[2], task[3], task[4], task[5]

    results = []
    
    for i in range(2):
        x11 = x1 - (1 - i)*x_size
        x22 = x11 + 2*x_size
        x11, x22 = round(x11, 3), round(x22, 3)

        for j in range(2):
            y11 = y1 - (1 - j)*y_size
            y22 = y11 + 2*y_size
            y11, y22 = round(y11, 3), round(y22, 3)

            #print(x11, y11, x22, y22)
            if x11 < 0 or y11 < 0 or x22 > 10 or y22 > 10:
                continue

            df_train_cell2 = df_train[(df_train.x >= x11) & (df_train.x <= x22) & (df_train.y >= y11) & (df_train.y <= y22)]
            df_test_cell2 = df_test[(df_test.x >= x11) & (df_test.x <= x22) & (df_test.y >= y11) & (df_test.y <= y22)]

            result = do_cell((df_train_cell2, df_test_cell2, x11, y11))
            results.append(result)

    slices = []   
    for x in results:
        slices.append(np.column_stack((x[1], x[2])))

    results = merge2(slices)

    df_results = pd.DataFrame(data = results[:, 1], index = [int(x) for x in results[:, 0]])
    results = pd.merge(df_results, df_test_cell, left_index=True, right_index=True)

    return np.column_stack((results.index, results[0].values))

def do_grid(df_train, df_test):
    tasks = queue.Queue()

    for i in range(n_cells):
        if do_score == 1 and i == n_cells - 1:
            continue

        x1 = i*x_size
        if do_cv == 1 or grid_search == 1:
            x2 = x1 + x_size
        else:
            x2 = x1 + 2*x_size

        x1, x2 = round(x1, 3), round(x2, 3)

        if not (x1 >= x_min and x2 <= x_max):
            continue

        if x2 == 10:
            x2 += 0.1

        df_train_x = df_train[(df_train.x >= x1) & (df_train.x < x2)]
        df_test_x = df_test[(df_test.x >= x1) & (df_test.x < x2)]
 
        for j in range(n_cells):
            if do_score == 1 and j == n_cells - 1:
                continue

            if (do_cv == 1 or grid_search == 1) and (n_cells*i + j)%skip_in_cv != cv_start_at:
                continue

            y1 = j*y_size;
            if do_cv == 1 or grid_search == 1:
                y2 = y1 + y_size
            else:
                y2 = y1 + 2*y_size;

            y1, y2 = round(y1, 3), round(y2, 3)
      
            if y2 == 10:
                y2 += 0.1

            #print(x1, x2, y1, y2)
            df_train_xy = df_train_x[(df_train_x.y >= y1) & (df_train_x.y < y2)]
            df_test_xy = df_test_x[(df_test_x.y >= y1) & (df_test_x.y < y2)]
            
            if do_cv == 0 and grid_search == 0:
                tasks.put((df_train_xy, df_test_xy, x1, y1))
            else:
                tasks.put((df_train, df_test, df_train_xy, df_test_xy, x1, y1))

            if tasks.qsize() % 100 == 0:
                print('tasks', tasks.qsize())

            if tasks.qsize() >= test_max_cells:
                break

        if tasks.qsize() >= test_max_cells:
            break

    if do_cv == 0 and grid_search == 0:
        results = run_tasks(tasks, do_cell, n_threads)
    else:
        results = run_tasks(tasks, do_cell_cv, n_threads)
 
    slices = []   
    if do_cv == 0 and grid_search == 0:
        for x in results:
            slices.append(np.column_stack((x[1], x[2])))
        results = merge2(slices)
    else:
        results = merge2(results)

    return results

if __name__ == '__main__':
    np.random.seed(0)

    points = [(4.8,9.1),(0.5,2.9),(7.2,4.2),(1.7,1.1),(8.2,4.7),(8.8,7.3),(4.1,3.7),(5.2,8.9),(7.4,0.1),(6.9,9.2),(7.1,1.8),(4.8,1.4),(3.6,9.4),(9.2,2.8),(3.4,6.0),(9.6,1.5),(2.6,8.7),(4.9,9.0),(1.9,5.3),(3.3,3.2),(4.5,4.3),(3.6,9.1),(7.3,7.3),(2.9,5.8),(7.8,8.0),(3.4,7.7),(7.4,1.4),(8.7,4.4),(4.9,4.5),(5.7,6.2),(5.0,8.7),(6.3,4.0),(4.2,8.1),(3.5,2.1),(0.6,8.8),(9.2,1.2),(3.3,1.8),(1.2,9.0),(0.6,9.8),(1.0,8.6),(5.7,3.7),(3.4,7.6),(3.1,6.6),(5.2,4.8),(9.0,5.5),(8.3,7.3),(0.4,7.7),(2.2,9.0),(0.4,3.3),(1.0,4.8),(8.2,3.0),(1.5,3.3),(8.1,1.4),(2.3,0.7),(7.1,4.0),(3.1,7.2),(3.4,7.3),(8.2,2.2),(9.7,1.6),(2.9,1.8),(3.5,4.8),(5.2,8.5),(8.9,2.2),(6.2,1.1),(4.6,3.2),(3.2,4.8),(7.3,0.7),(8.8,7.3),(1.8,9.4),(2.0,5.3),(2.9,3.0),(5.9,9.2),(8.1,7.2),(5.6,9.2),(4.9,8.7),(8.3,2.1),(7.7,0.1),(3.2,2.3),(5.1,7.4),(1.0,5.1),(9.4,2.3),(6.8,5.9),(0.1,4.8),(7.1,0.4),(8.8,5.2),(0.3,2.2),(9.5,5.8),(1.1,2.9),(4.6,0.2),(4.1,4.9),(2.4,5.9),(7.5,2.4),(6.2,6.4),(9.5,7.8),(8.5,4.9),(1.3,4.7),(0.7,9.4),(9.6,7.2),(3.5,2.5),(2.7,1.3)]

    if len(sys.argv) == 1:
        one_cell = 1
    elif sys.argv[1] == 'x':
        few_cells = 1
    elif sys.argv[1] == 'g':
        grid_search = 1
    elif sys.argv[1] == 'c':
        do_cv = 1
        if len(sys.argv) >= 3:
            x_min = float(sys.argv[2])
            x_max = float(sys.argv[3])
        print(x_min, x_max)
    elif sys.argv[1] == 's':
        do_score = 1
        if len(sys.argv) >= 3:
            x_min = float(sys.argv[2])
            x_max = float(sys.argv[3])
        print(x_min, x_max)
    else:
        raise ValueError ('***Unexpected***')

    #Loading data
    train_file = 'Dataset\\train.csv'
    test_file = 'Dataset\\test.csv'

    print('load data', datetime.datetime.now())
    df_train = pd.read_csv(train_file, index_col = 0)
    print('load data', datetime.datetime.now())
    df_test = pd.read_csv(test_file, index_col = 0)

    if split_by_time == 1 or full_dataset == 0:
        train_len = int(df_train.shape[0]*0.8)
        df_train_sorted = df_train.sort_values('time')
        train80 = df_train_sorted[:train_len].sort_index().index
        train20 = df_train_sorted[train_len:].sort_index().index
        
    if full_dataset == 0:
        df_train = df_train_sorted[:train_len].sort_index()
        df_test = df_train_sorted[train_len:].sort_index()
 
    df_train = fe_2(df_train)
    df_test = fe_2(df_test)

    '''
    ##New feature x/y
    df_train['x_d_y'] = df_train.x.values / (df_train.y.values + eps) 
    df_test['x_d_y'] = df_test.x.values / (df_test.y.values + eps) 
    ##New feature x*y
    df_train['x_t_y'] = df_train.x.values * df_train.y.values  
    df_test['x_t_y'] = df_test.x.values * df_test.y.values
    '''

    print(df_train.shape)
    print(df_test.shape)

    if one_cell == 1:
        x1, y1 = points[0][0], points[0][1]
        
        df_train_cell, df_test_cell = get_cell(df_train, df_test, x1, y1)
        do_cell((df_train_cell, df_test_cell, x1, y1))

    if few_cells == 1:
        #np.random.seed(4242)
        #np.random.seed(1234)

        n_cells = 100
        x_size = 10/n_cells
        y_size = 10/n_cells
        n_places = 100
     
        points = points[:50]

        tasks = queue.Queue()

        for x1, y1 in points:
            x2 = round(x1 + x_size, 2)
            y2 = round(y1 + y_size, 2)
            tasks.put((x1, y1))

        #results = run_tasks(tasks, do_cell_cv_1)

        score = np.array(results).mean()
        print('score', score)
  
    if do_cv == 1:
        #80% train, 20% test
        if split_by_time == 1:
            df_train_fold = df_train.iloc[train80]
            df_test_fold = df_train.iloc[train20]
        else:
            #use KFold; train_test_split doesn't return arrays sorted by row_id
            folds = KFold(df_train.shape[0], n_folds=5, shuffle=True)
            for train, test in folds:
                df_train_fold = df_train.iloc[train]
                df_test_fold = df_train.iloc[test]
                break
    
        print(df_train_fold.shape)
        print(df_test_fold.shape)

        #seed 0 to compare with grid_search
        np.random.seed(0)
        results = do_grid(df_train_fold, df_test_fold)
   
        scores = []
        result_index = 0
        for i in range(len(df_test_fold.index)):
            if result_index < len(results) and df_test_fold.index[i] == int(results[result_index][0]):
                dict = json.loads(results[result_index][1])
                dict2 = {}
                for key, value in dict.items():
                    dict2[int(key)]=float(value)
                dict = dict2
                
                top3 = sorted(dict, key=dict.__getitem__, reverse=True)[:3]
                score = calculate_score_per_row2(top3, df_test_fold.place_id[df_test_fold.index[i]])
                scores.append(score)

                result_index += 1

        score = np.array(scores).mean()
        print('score', score, len(scores))

    if grid_search == 1:
        #test_max_cells = 3
        xgb_calculate_n_estimators = False
        n_estimators_fixed = 100

        n_cells_list = [100]
        #n_places_th_list = [6, 8, 10, 12]
        n_places_th_list = [5]

        if split_by_time == 1:
            df_train_fold = df_train.iloc[train80]
            df_test_fold = df_train.iloc[train20]
        else:
            #use KFold; train_test_split doesn't return arrays sorted by row_id
            folds = KFold(df_train.shape[0], n_folds=5, shuffle=True)
            for train, test in folds:
                df_train_fold = df_train.iloc[train]
                df_test_fold = df_train.iloc[test]
                break
    
        print(df_train_fold.shape)
        print(df_test_fold.shape)

        all_scores = []
        for n_cells in n_cells_list:
            x_size = 10/n_cells
            y_size = 10/n_cells

            #for foo in [1]:
            for n_places_th in n_places_th_list:
                np.random.seed(0)
        
                start_time = datetime.datetime.now()  
                results = do_grid(df_train_fold, df_test_fold)
                duration = (datetime.datetime.now() - start_time).seconds//60
   
                scores = []
                result_index = 0
                for i in range(len(df_test_fold.index)):
                    if result_index < len(results) and df_test_fold.index[i] == int(results[result_index][0]):
                        dict = json.loads(results[result_index][1])
                        dict2 = {}
                        for key, value in dict.items():
                            dict2[int(key)]=float(value)
                        dict = dict2
                        
                        top3 = sorted(dict, key=dict.__getitem__, reverse=True)[:3]
                        score = calculate_score_per_row2(top3, df_test_fold.place_id[df_test_fold.index[i]])
                        scores.append(score)

                        result_index += 1

                score = np.array(scores).mean()
               
                result = [score, n_cells, duration]
                print(str(result).replace(',', '').replace('[', '').replace(']', ''))
                all_scores.append(result)
               
        for x in all_scores:
            print(str(x).replace(',', '').replace('[', '').replace(']', ''))

    if do_score == 1:
        results = do_grid(df_train, df_test)

        sub = pd.DataFrame(results, columns=['row_id', 'place_id'])
        sub.to_csv('results_' + str(x_min) + '.csv', index=False)            

    print('done', datetime.datetime.now())
