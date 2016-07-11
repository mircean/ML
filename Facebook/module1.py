import sys
import datetime
import numpy as np
import pandas as pd
import queue 

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from xgboost.sklearn import XGBClassifier
from xgboost.core import DMatrix
from xgboost.training import train, cv

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from common import merge, merge2, run_tasks

###consts
eps = 0.00001  #required to avoid some divisions by zero.

###parameters
split_by_time = 1
full_dataset = 1
n_threads = 10
n_topx = 7

###for testing
test_max_cells = 10000000
skip_in_cv = 11
cv_start_at = 5

##features
xy = 0

###for cells/size
n_cells = 75
x_size = 10/n_cells
y_size = 10/n_cells

radius = 0

extra_train = 0.5

n_places = 0
n_places_th = 10
n_places_percentage = 0.9

### for clf
train_test = 1
n_folds = 5

classifier = 'xgb'

xgb_calculate_n_estimators = False
n_estimators_fixed = 100
learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.1, 6, 0.5, 0.5, 0, 1, 1, 0
early_stopping_rounds = 20

rf_n_estimators = 1000 #300

fw = [500., 1000., 4., 3., 2., 8.] #feature weights (black magic here)

###done parameters

one_cell = 0
few_cells = 0
do_test = 0
grid_search = 0
do_cv = 0
do_score = 0
x_min, x_max = 0, 10

def fe_time_0(df):
    df['min']=df.time%60
    df['hour']=df.time//60%24
    df['day']=df.time//(60*24)

    df = df.drop(['time'], axis=1)
    return df

def fe_time_1(df):
    #df['min']=df.time%60
    df['hour']=df.time//60%24 + (df.time%60)/60
    df['day']=df.time//(60*24)
    df['weekday'] = df.time//(60*24)%7
    df['month'] = df['time']//(60*24*30)%12 #rough estimate, month = 30 days
    df['year'] = df['time']//(60*24*30*12)

    df = df.drop(['time'], axis=1)

    if xy == 1:
    ##New feature x/y
        df_train['x_d_y'] = df_train.x.values / (df_train.y.values + eps) 
        df_test['x_d_y'] = df_test.x.values / (df_test.y.values + eps) 
        ##New feature x*y
        df_train['x_t_y'] = df_train.x.values * df_train.y.values  
        df_test['x_t_y'] = df_test.x.values * df_test.y.values
    
    return df

def fe_time_3(df):
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]') 
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') for mn in df.time.values)    
    df['hour'] = d_times.hour + d_times.minute/60
    df['weekday'] = d_times.weekday
    df['day'] = d_times.day
    df['month'] = d_times.month
    df['year'] = d_times.year - 2014

    df = df.drop(['time'], axis=1)
    return df

def fe_4(df):
    df['hour'] = df.time//60%24 + 1 + (df.time%60)/60  
    df['day'] = df.time//(60*24)
    df['weekday'] = df.time//(60*24)%7
    df['month'] = df['time']//(60*24*30)%12 #rough estimate, month = 30 days
    df['year'] = df['time']//(60*24*30*12)
    df = df.drop(['time'], axis=1)

    #df['accuracy'] = np.log10(df['accuracy'])

    return df

def fe_5(df):
    df['hour'] = df.time//60%24 + (df.time%60)/60
    #df['day'] = df.time//(60*24)%365
    df['weekday'] = df.time//(60*24)%7
    df['month'] = df['time']//(60*24*30)%12 #rough estimate, month = 30 days
    #df['month'] = df['day']//30.416%12 
    df['year'] = df['time']//(60*24*365)
    df = df.drop(['time'], axis=1)

    return df

def fe_6(df):
    #fw = [500., 1000., 4., 3., 2., 8.] #feature weights (black magic here)
    pd.options.mode.chained_assignment = None

    df['hour'] = (df['hour'] + 1)*fw[2]
    #df['day'] = df['day'] + 1
    df['weekday'] = (df['weekday'] + 1)*fw[3]
    df['month'] = (df['month'] + 1)*fw[4]
    df['year'] = (df['year'] + 1)*fw[5]

    pd.options.mode.chained_assignment = 'warn'

    return df

def fe_circular_hour(df):
    pd.options.mode.chained_assignment = None

    add_data = df[df.hour <= 2]
    add_data2 = df[df.hour >= 24]

    add_data.hour = add_data.hour + 24
    df = df.append(add_data)

    add_data2.hour = add_data2.hour - 24
    df = df.append(add_data2)

    pd.options.mode.chained_assignment = 'warn'

    print('fe_circular_hour', df.shape[0])
    return df

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

def get_cell(df_train, df_test, x1, y1):
    x2, y2 = round(x1 + x_size, 2), round(y1 + y_size, 2)

    if extra_train >= 1.01:
        raise ValueError("extra_train too big")
        
    if radius == 0:
        df_train_cell = df_train[(df_train.x >= x1 - extra_train*x_size/2) & (df_train.x < x2 + extra_train*x_size/2) & (df_train.y >= y1 - extra_train*y_size/2) & (df_train.y < y2 + extra_train*y_size/2)]
    else:
        xc, yc = round(x1 + x_size/2, 2), round(y1 + y_size/2, 2)
        df_train_cell = df_train[pow((df_train.x - xc), 2) + pow((df_train.y - yc), 2) <= pow(radius, 2)]

    df_test_cell = df_test[(df_test.x >= x1) & (df_test.x < x2) & (df_test.y >= y1) & (df_test.y < y2)]

    #print(df_train_cell.shape, df_test_cell.shape)
    return df_train_cell, df_test_cell

def do_cell(task):
    df_train, df_test, x_start, y_start = task[0], task[1], task[2], task[3]
    #print('do_cell', df_train.shape, df_test.shape, x_start, y_start)

    #train
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

    if classifier == 'knn':
        df_train.loc[:,'x'] *= fw[0]
        df_train.loc[:,'y'] *= fw[1]
        df_test.loc[:,'x'] *= fw[0]
        df_test.loc[:,'y'] *= fw[1]
            
        df_train = fe_6(df_train)
        df_test = fe_6(df_test)

    X = df_train.drop(['place_id'], axis=1).values
    X_predict = df_test.values

    score = 0
    n_estimators = 0
    if classifier == 'xgb':    
        if xgb_calculate_n_estimators == True:
            clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='multi:softprob', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)

            if train_test == 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
                clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=calculate_score, early_stopping_rounds=early_stopping_rounds, verbose=10 if one_cell == 1 else False)
                score = round(1 - clf.booster().best_score, 6)
                n_estimators = int(1.25*clf.booster().best_ntree_limit)
            else:
                xgb_options = clf.get_xgb_params()
                xgb_options['num_class'] = n_places_local
                train_dmatrix = DMatrix(X, label=y)

                #some of the classes have less than n_folds, cannot use stratified KFold
                #folds = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
                folds = KFold(len(y), n_folds=n_folds, shuffle=True)
                cv_results = cv(xgb_options, train_dmatrix, clf.n_estimators, early_stopping_rounds=early_stopping_rounds, verbose_eval=10 if one_cell == 1 else False, show_stdv=False, folds=folds, feval=calculate_score)

                n_estimators = int(1.25*cv_results.shape[0])
                score = round(1 - cv_results.values[-1][0], 6)
                std = round(cv_results.values[-1][1], 6)
        else:
            n_estimators = n_estimators_fixed

        clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective='multi:softprob', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)
    elif classifier == 'rf':
        clf = RandomForestClassifier(n_estimators = rf_n_estimators, n_jobs = -1)
    elif classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=int(np.floor(np.sqrt(y.size)/5)), weights='distance', metric='manhattan', n_jobs=-1)
    else:
        abc += 1
    #if few_cells == 1 or grid_search == 1:
    #    return [score, None, None]

    clf.fit(X, y)
    y_predict = clf.predict_proba(X_predict)
    labels_predict = le.inverse_transform(np.argsort(y_predict, axis=1)[:,::-1][:,:n_topx])    

    print(x_start, y_start, score, n_estimators, n_places_local, n_places_th_local, percentage)

    return [score, row_ids, labels_predict]

def do_grid(df_train, df_test):
    tasks = queue.Queue()
    
    if extra_train >= 1.01:
        raise ValueError("extra_train too big")
        
    for i in range(n_cells):
        x_start = i*x_size
        x_end = x_start + x_size

        x_start, x_end = round(x_start, 2), round(x_end, 2)

        if not (x_start >= x_min and x_end <= x_max):
            continue

        if i == n_cells - 1:
            x_end += 0.1

        if radius == 0:
            df_train_x = df_train[(df_train.x >= x_start - extra_train*x_size/2) & (df_train.x < x_end + extra_train*x_size/2)]
        else:
            df_train_x = df_train[(df_train.x >= x_start + x_size/2 - radius) & (df_train.x <= x_end + x_size/2 + radius)]
        
        df_test_x = df_test[(df_test.x >= x_start) & (df_test.x < x_end)]
 
        for j in range(n_cells):
            if (do_cv == 1 or grid_search == 1) and (n_cells*i + j)%skip_in_cv != cv_start_at:
                continue

            y_start = j*y_size;
            y_end = y_start + y_size;

            y_start, y_end = round(y_start, 2), round(y_end, 2)

            if j == n_cells - 1:
                y_end += 0.1

            #ignore cells close to edges
            #if test_max_cells <= 1000 and (do_cv == 1 or grid_search == 1) and not (x_start + x_size/2 - radius >= 0 and x_start + x_size/2 + radius <= 10 and y_start + y_size/2 - radius >= 0 and y_start + y_size/2 + radius <= 10):
            #    print('Ignore cell', x_start, y_start)
            #    continue

            #print(x_start, x_end, y_start, y_end)
            if radius == 0:
                df_train_xy = df_train_x[(df_train_x.y >= y_start - extra_train*y_size/2) & (df_train_x.y < y_end + extra_train*y_size/2)]
            else:
                xc, yc = round(x_start + x_size/2, 2), round(y_start + y_size/2, 2)
                df_train_xy = df_train_x[pow((df_train_x.x - xc), 2) + pow((df_train_x.y - yc), 2) <= pow(radius, 2)]

            df_test_xy = df_test_x[(df_test_x.y >= y_start) & (df_test_x.y < y_end)]

            tasks.put((df_train_xy, df_test_xy, x_start, y_start))

            if tasks.qsize() % 100 == 0:
                print('tasks', tasks.qsize())

            if tasks.qsize() >= test_max_cells:
                break

        if tasks.qsize() >= test_max_cells:
            break

    results = run_tasks(tasks, do_cell, n_threads = n_threads)
    return results

if __name__ == '__main__':
    np.random.seed(0)

    if len(sys.argv) == 1:
        one_cell = 1
    elif sys.argv[1] == 'x':
        few_cells = 1
    elif sys.argv[1] == 't':
        do_test = 1
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

    points = [(4.8,9.1),(0.5,2.9),(7.2,4.2),(1.7,1.1),(8.2,4.7),(8.8,7.3),(4.1,3.7),(5.2,8.9),(7.4,0.1),(6.9,9.2),(7.1,1.8),(4.8,1.4),(3.6,9.4),(9.2,2.8),(3.4,6.0),(9.6,1.5),(2.6,8.7),(4.9,9.0),(1.9,5.3),(3.3,3.2),(4.5,4.3),(3.6,9.1),(7.3,7.3),(2.9,5.8),(7.8,8.0),(3.4,7.7),(7.4,1.4),(8.7,4.4),(4.9,4.5),(5.7,6.2),(5.0,8.7),(6.3,4.0),(4.2,8.1),(3.5,2.1),(0.6,8.8),(9.2,1.2),(3.3,1.8),(1.2,9.0),(0.6,9.8),(1.0,8.6),(5.7,3.7),(3.4,7.6),(3.1,6.6),(5.2,4.8),(9.0,5.5),(8.3,7.3),(0.4,7.7),(2.2,9.0),(0.4,3.3),(1.0,4.8),(8.2,3.0),(1.5,3.3),(8.1,1.4),(2.3,0.7),(7.1,4.0),(3.1,7.2),(3.4,7.3),(8.2,2.2),(9.7,1.6),(2.9,1.8),(3.5,4.8),(5.2,8.5),(8.9,2.2),(6.2,1.1),(4.6,3.2),(3.2,4.8),(7.3,0.7),(8.8,7.3),(1.8,9.4),(2.0,5.3),(2.9,3.0),(5.9,9.2),(8.1,7.2),(5.6,9.2),(4.9,8.7),(8.3,2.1),(7.7,0.1),(3.2,2.3),(5.1,7.4),(1.0,5.1),(9.4,2.3),(6.8,5.9),(0.1,4.8),(7.1,0.4),(8.8,5.2),(0.3,2.2),(9.5,5.8),(1.1,2.9),(4.6,0.2),(4.1,4.9),(2.4,5.9),(7.5,2.4),(6.2,6.4),(9.5,7.8),(8.5,4.9),(1.3,4.7),(0.7,9.4),(9.6,7.2),(3.5,2.5),(2.7,1.3)]

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

    df_train = fe_5(df_train)
    df_test = fe_5(df_test)

    print(df_train.shape)
    print(df_test.shape)

    if one_cell == 1:
        xgb_calculate_n_estimators = True
        n_estimators_fixed = 100
        
        n_cells = 75
        x_size = 10/n_cells
        y_size = 10/n_cells
        
        extra_train = 0.1
    
        #x1, y1 = points[0][0], points[0][1]
        x1, y1 = 0, 2.5
        df_train_cell, df_test_cell = get_cell(df_train, df_test, x1, y1)
        do_cell((df_train_cell, df_test_cell, x1, y1))

    if few_cells == 1:
        #np.random.seed(4242)
        #np.random.seed(1234)
        xgb_calculate_n_estimators = True
        n_estimators_fixed = 1
        
        n_cells = 50
        x_size = 10/n_cells
        y_size = 10/n_cells
        
        extra_train = 0.1

        one_cell = 1 #for logging
        for x1, y1 in points[:3]:
            df_train_cell, df_test_cell = get_cell(df_train, df_test, x1, y1)
            do_cell((df_train_cell, df_test_cell, x1, y1))
   
        '''
        tasks = queue.Queue()

        for x1, y1 in points[:3]:
            df_train_cell, df_test_cell = get_cell(df_train, df_test, x1, y1)
            tasks.put((df_train_cell, df_test_cell, x1, y1))

        results = run_tasks(tasks, do_cell)
        '''
        '''
        scores = []
        for x in results:
            scores.append(x[0])

        score = np.array(scores).mean()
        print('score', score)
        '''
    if do_test == 1:
        '''
        n_cells = 100
        radius = 0.24

        #xc, yc = points[0][0], points[0][1]
        xc, yc = points[2][0], points[2][1]
        folds = KFold(df_train.shape[0], n_folds=5, shuffle=True)
        for train, test in folds:
            df_train_fold = df_train.iloc[train]
            df_test_fold = df_train.iloc[test]
            break

        df_train_cell = df_train_fold[pow((df_train_fold.x - xc), 2) + pow((df_train_fold.y - yc), 2) <= pow(radius, 2)]
        df_test_cell = df_test_fold[pow((df_test_fold.x - xc), 2) + pow((df_test_fold.y - yc), 2) <= pow(radius, 2)]
        
        results = do_cell((df_train_cell, df_test_cell, xc, yc))
        
        for size in [0.33, 0.2, 0.1, 0.05, 0.04]:
            x1, y1 = xc - size/2, yc - size/2
            x2, y2 = xc + size/2, yc + size/2

            #df_train_cell_2 = df_test_cell[(df_test_cell.x >= x1) & (df_test_cell.x < x2) & (df_test_cell.y >= y1) & (df_test_cell.y < y2)]

            #plt.plot(df_test_cell.x, df_test_cell.y, 'bo')

            a1, a2, a3, a4 = [], [], [], []
            scores = []
            for i in range(df_test_cell.shape[0]):
                if df_test_cell.iloc[i].x >= x1 and df_test_cell.iloc[i].x < x2 and df_test_cell.iloc[i].y >= y1 and df_test_cell.iloc[i].y < y2:
                    score = calculate_score_per_row2(results[2][i], df_test_cell.iloc[i].place_id)
                    scores.append(score)
                    if score == 1:
                        a1.append([df_test_cell.iloc[i].x, df_test_cell.iloc[i].y])
                    elif score == 0.5:
                        a2.append([df_test_cell.iloc[i].x, df_test_cell.iloc[i].y])
                    elif round(score, 2) == 0.33:
                        a3.append([df_test_cell.iloc[i].x, df_test_cell.iloc[i].y])
                    else:
                        a4.append([df_test_cell.iloc[i].x, df_test_cell.iloc[i].y])

            score = np.array(scores).mean()
            print(size, score, len(scores))

            a1 = np.array(a1)
            a2 = np.array(a2)
            a3 = np.array(a3)
            a4 = np.array(a4)

            #plt.plot(a1[:,0], a1[:,1], 'bo')
            #plt.plot(a2[:,0], a2[:,1], 'ro')
            #plt.plot(a3[:,0], a3[:,1], 'yo')
            #plt.plot(a4[:,0], a4[:,1], 'ko')
            #plt.show()
        '''
        for radius in [0.04/pow(2, 1/2), 0.05/pow(2, 1/2), 0.1/pow(2, 1/2), 0.2/pow(2, 1/2), 0.5/pow(2, 1/2), 1/pow(2, 1/2)]:
            print('radius', radius)
            xc, yc = 3, 4

            folds = KFold(df_train.shape[0], n_folds=5, shuffle=True)
            for train, test in folds:
                df_train_fold = df_train.iloc[train]
                df_test_fold = df_train.iloc[test]
                break

            df_train_cell = df_train_fold[pow((df_train_fold.x - xc), 2) + pow((df_train_fold.y - yc), 2) <= pow(radius, 2)]
            df_test_cell = df_test_fold[pow((df_test_fold.x - xc), 2) + pow((df_test_fold.y - yc), 2) <= pow(radius, 2)]

            results = do_cell((df_train_cell, df_test_cell, xc, yc))

            scores = []
            for i in range(df_test_cell.shape[0]):
                score = calculate_score_per_row2(results[2][i], df_test_cell.iloc[i].place_id)
                scores.append(score)

            score = np.array(scores).mean()
            print(score, len(scores), df_test_cell.shape[0])

    if do_cv == 1:
        '''
        n_cells = 75
        x_size = 10/n_cells
        y_size = 10/n_cells
        '''
        '''
        extra_train = 0.1
        n_places_th = 0
        xgb_calculate_n_estimators = True
        train_test = 0
        '''
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

        #to compare with grid_search
        np.random.seed(0)
        results = do_grid(df_train_fold, df_test_fold)

        scores = []
        for x in results:
            #x[1] = index, x[2] = preds
            for i in range(len(x[1])):
                score = calculate_score_per_row2(x[2][i], df_test_fold.place_id[x[1][i]])
                scores.append(score)

        score = np.array(scores).mean()
        print('score', score, len(scores))

    if grid_search == 1:
        xgb_calculate_n_estimators = False
        n_estimators_fixed = 100
        
        n_cells_list = [75]
        extra_train_list = [0.1]
        
        n_places_percentage = 0.9
        n_places_th = 0
        #n_places_th_list = [10]
        #n_places_percentage_list = [0.925, 0.95]

        if split_by_time == 1:
            df_train_fold = df_train.iloc[train80]
            df_test_fold = df_train.iloc[train20]
        else:
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

            for extra_train in extra_train_list:
                #for n_places_th in n_places_th_list:
				#for n_places_percentage in n_places_percentage_list:
                #for n_estimators_fixed in [160]:
                for foo in [1]:
                    np.random.seed(0)

                    start_time = datetime.datetime.now()
                    results = do_grid(df_train_fold, df_test_fold)
                    duration = (datetime.datetime.now() - start_time).seconds//60

                    scores = []
                    for x in results:
                        #x[1] = index, x[2] = preds
                        for i in range(len(x[1])):
                            score = calculate_score_per_row2(x[2][i], df_test_fold.place_id[x[1][i]])
                            scores.append(score)

                    score = np.array(scores).mean()
                    print('score', score, len(scores))

                    result = [score, n_cells, extra_train, duration]
                    #result = [score, n_cells, radius, n_places, n_places_th, n_places_percentage, duration]
                    print(str(result).replace(',', '').replace('[', '').replace(']', ''))
                    all_scores.append(result)

        for x in all_scores:
            print(str(x).replace(',', '').replace('[', '').replace(']', ''))

    if do_score == 1:      
        results = do_grid(df_train, df_test)

        slices = []
        for x in results:
            slices.append(np.column_stack((x[1], x[2])))
           
        results = merge(slices)
        print('results merged', results.shape[0], datetime.datetime.now())

        sub = pd.DataFrame(results, columns=['row_id', 'place_id'])
        sub.to_csv('results_' + str(x_min) + '.csv', index=False)            

    print('done', datetime.datetime.now())