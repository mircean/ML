import datetime
import numpy as np
import pandas as pd
import queue 

from sklearn.cross_validation import KFold

from common import run_tasks

n_topx = 7 #was 3
knn = 0
m12 = 0
m12_rf = 0
m12_rf_knn = 0
scores = 1
all = 1
grid_search = 0

static_scores = np.array([round(1/(i+1),3) for i in range(n_topx)])

def calculate_score_per_row2(y_predicted, y_true):
    score = 0
    for i in range(3):
        if(int(y_predicted[i]) == y_true):
            score = 1/(i + 1)
            break
    return score

print('load data', datetime.datetime.now())
df_train = pd.read_csv('Dataset/train.csv', index_col = 0)
train_len = int(df_train.shape[0]*0.8)
df_train_sorted = df_train.sort_values('time')
df_test = df_train_sorted[train_len:].sort_index()


filter = [x for x in range(df_test.shape[0])]
'''
np.random.seed(0)
folds = KFold(df_test.shape[0], n_folds = 2, shuffle = True)
for train, test in folds:
    filter = train    
    break
'''

df_test = df_test.iloc[filter]
test = np.column_stack((df_test.index, df_test['place_id'].values))
del df_test

files = []
if knn == 1:
    file = pd.read_csv('ensemble_test/knn.cs')
    files.append(np.copy(file.iloc[filter].values))
    del file

    file = pd.read_csv('ensemble_test/knn_v2.csv')
    files.append(np.copy(file.iloc[filter].values))
    del file

if m12 == 1:
    file = pd.read_csv('ensemble_test/m1.csv')
    files.append(np.copy(file.iloc[filter].values))
    del file

    file = pd.read_csv('ensemble_test/m2.csv')
    files.append(np.copy(file.iloc[filter].values))
    del file

if m12_rf == 1:
    file = pd.read_csv('ensemble_test/rf.csv')
    files.append(np.copy(file.iloc[filter].values))
    del file

    file = pd.read_csv('ensemble_test/l2_m12.csv')
    files.append(np.copy(file.iloc[filter].values))
    del file

if m12_rf_knn == 1:
    file = pd.read_csv('ensemble_test/l2_knn.csv')
    files.append(np.copy(file.iloc[filter].values))
    del file

    file = pd.read_csv('ensemble_test/l3_m12_rf.csv')
    files.append(np.copy(file.iloc[filter].values))
    del file

if all == 1:
    #file = pd.read_csv('ensemble_test/all.csv')
    file = pd.read_csv('ensemble_test/all_guess_weights3.csv')
    files.append(np.copy(file.iloc[filter].values))
    del file

print('files read', len(files))

if scores == 1:
    scores = []
    for i in range(files[0].shape[0]):  
        scores_per_model = []
        row_id = test[i][0]
        for j in range(len(files)):
            if row_id != files[j][i][0]:
                abc += 1
            places = files[j][i][1]
        
            score_per_model = calculate_score_per_row2(places.split()[:3], test[i][1])
            scores_per_model.append(score_per_model)
 
        scores.append(scores_per_model)

    scores_per_model = np.array(scores)
    print('Score per models', [scores_per_model[:, i].mean() for i in range(len(files))])

if grid_search == 1:
    w0_min = 48
    w0_max = 48
    w0_rate = 1

    '''
    m1_min = 32
    m1_max = 34
    m1_rate = 2
    '''

    tasks = queue.Queue()

    for i in range(1 + int((w0_max - w0_min)/w0_rate)):
        w0 = (w0_min + i*w0_rate)/100
        w1 = 1 - w0
        weights = np.array([round(w0, 3), round(w1, 3)])
        
        tasks.put((weights, 1))

        '''
        others = 1 - w0
   
        for j in range(1 + int((m1_max - m1_min)/m1_rate)):
            w1 = (m1_min + j*m1_rate)/100*others
            w2 = others - w1
        
            weights = np.array([round(w0, 3), round(w1, 3), round(w2, 3)])
        
            tasks.put((weights, 1))
            #if tasks.qsize() > 2:
            #    break
       
        '''
    def task_fn(task):
        weights = task[0]

        scores = []
        for i in range(files[0].shape[0]):  
        #for i in range(2):
            row_id = test[i][0]
            dict = {}
            for j in range(len(files)):
                places = files[j][i][1]
        
                for place_id, score in zip(places.split(), static_scores*weights[j]):
                    if dict.get(place_id) != None:
                        dict[place_id] += score
                    else:
                        dict[place_id] = score   

            #print(dict)
            top3 = sorted(dict, key=dict.__getitem__, reverse=True)[:3]
            #print(top3)
            score = calculate_score_per_row2(top3, test[i][1])
            scores.append(score)

            #if i % 1000000 == 10:
            #    print(i)
            #    print(scores[i-10:i])

        score = np.array(scores).mean()
        print(score, weights)

    run_tasks(tasks, task_fn, n_threads = 1)
