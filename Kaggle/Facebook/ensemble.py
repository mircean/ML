import datetime
import numpy as np
import pandas as pd

n_topx_in = 7 #was 3
static_scores = np.array([round(1/(i+1),3) for i in range(n_topx_in)])

n_topx_out = 3

slices = []
'''
#e1
slices.append(pd.read_csv('m1_s14.csv').values)
slices.append(pd.read_csv('m2_s11.csv').values)
slices.append(pd.read_csv('knn_58018.csv').values)

weights = [0.3, 0.22, 0.48]
'''
'''
#e2
slices.append(pd.read_csv('m1_58036.csv').values)
slices.append(pd.read_csv('m2_59115.csv').values)
slices.append(pd.read_csv('knn_58143.csv').values)
weights = [0.2, 0.32, 0.48]
'''
'''
#e3 - top7
slices.append(pd.read_csv('m1.csv').values)
slices.append(pd.read_csv('m2.csv').values)
slices.append(pd.read_csv('knn_58143.csv').values)
#e3 - guess, best so far
#weights = [0.2, 0.32, 0.48]
#e31 - grid search gave low weight to knn bc knn doesn't do well on test0
#weights = [0.327, 0.433, 0.24]
#e32 - grid search keeping knn=0.48 gave very low weight to m1
weights = [0.036, 0.484, 0.48]

#maybe knn overfits?
'''

'''
#e4 - was best
slices.append(pd.read_csv('m1.csv').values)
slices.append(pd.read_csv('m2.csv').values)
slices.append(pd.read_csv('knn_58177.csv').values)
weights = [0.2, 0.32, 0.48]

#e41
slices.append(pd.read_csv('m1.csv').values)
slices.append(pd.read_csv('m2.csv').values)
slices.append(pd.read_csv('knn_58177.csv').values)
weights = [0.33, 0.67, 0]

#e42
slices.append(pd.read_csv('m1.csv').values)
slices.append(pd.read_csv('m2.csv').values)
slices.append(pd.read_csv('knn_58177.csv').values)
weights = [0.172, 0.348, 0.48]

#e5
slices.append(pd.read_csv('m1.csv').values)
slices.append(pd.read_csv('m2.csv').values)
slices.append(pd.read_csv('knn_58177.csv').values)
slices.append(pd.read_csv('rf.csv').values)
weights = [0.30, 0.64, 0, 0.06]

#e51
slices.append(pd.read_csv('m1.csv').values)
slices.append(pd.read_csv('m2.csv').values)
slices.append(pd.read_csv('knn_58177.csv').values)
slices.append(pd.read_csv('rf.csv').values)
weights = [0.17, 0.29, 0.45, 0.06]

#e6 - best!!
slices.append(pd.read_csv('m1_58417.csv').values)
slices.append(pd.read_csv('m2_59279.csv').values)
slices.append(pd.read_csv('knn_58177.csv').values)
slices.append(pd.read_csv('knn_v2_58220_7.csv').values)
weights = [0.2, 0.32, 0.22, 0.26]

#run in Ensemblex folder for test file
#run in Ensemble_test for 20% test
#run with topx = 3 for submit, topx = 7 for test or input for next level ensemble

#e7.1
n_topx_out = 7
slices.append(pd.read_csv('knn_58177.csv').values)
slices.append(pd.read_csv('knn_v2_58220_7.csv').values)
weights = [0.49, 0.51]

#e7.2
#n_topx_out = 7
slices.append(pd.read_csv('m1_58417.csv').values)
slices.append(pd.read_csv('m2_59279.csv').values)
weights = [0.32, 0.68]
#weights = [0.49, 0.51]

#e7.3
n_topx_out = 7
slices.append(pd.read_csv('rf.csv').values)
slices.append(pd.read_csv('l2_m12.csv').values)
weights = [0.27, 0.73]

#e7.4
slices.append(pd.read_csv('l2_knn.csv').values)
slices.append(pd.read_csv('l3_m12_rf.csv').values)
#weights = [0.26, 0.74]
weights = [0.48, 0.52]
'''

#e7.5 - flat
slices.append(pd.read_csv('knn_58177.csv').values)
slices.append(pd.read_csv('knn_v2_58220_7.csv').values)
slices.append(pd.read_csv('knn_v3.csv').values)
#slices.append(pd.read_csv('knn.csv').values)
#slices.append(pd.read_csv('knn_v2.csv').values)
#slices.append(pd.read_csv('l2_knn.csv').values)

slices.append(pd.read_csv('m1_58417.csv').values)
slices.append(pd.read_csv('m1_v2.csv').values)
slices.append(pd.read_csv('m2_59279.csv').values)
slices.append(pd.read_csv('m2_v2.csv').values)
#slices.append(pd.read_csv('m1.csv').values)
#slices.append(pd.read_csv('m2.csv').values)

slices.append(pd.read_csv('rf_v2.csv').values)

slices.append(pd.read_csv('l2_m12.csv').values)

#weights = [0.22, 0.26, 0.2, 0.32, 0.0] #guess1
#weights = [0.12, 0.16, 0.20, 0.2, 0.32] #add knn3. best so far
#weights = [0.48, 0.2, 0.32] #only best knn, not good
#try1
#weights = [0.11, 0.15, 0.19, 0.19, 0.32, 0.04] #add rf. best so far
#try2
#weights = [0.10, 0.14, 0.18, 0.20, 0.33, 0.05] #take a bit from knn, give to m1, m2, rf. best so far
#try3
#weights = [0.10, 0.14, 0.18, 0.15, 0.25, 0.13, 0.05] #try2 + m2v2. best so far
#try4
#weights = [0.10, 0.14, 0.18, 0.12, 0.08, 0.23, 0.10, 0.05] #try3 + m1v2. Best!!!

#try5
#weights = [0.09, 0.13, 0.17, 0.13, 0.085, 0.24, 0.105, 0.05] #try4 + take a bit more from knn

#try6
weights = [0.095, 0.135, 0.175, 0.09, 0.06, 0.20, 0.08, 0.045, 0.12] #try4 + l2_m12


print('files read', len(slices))

start_time = datetime.datetime.now()  

places_merged = []
for i in range(slices[0].shape[0]):
#for i in range(2):
    dict = {}
    for j in range(len(slices)):
        places = slices[j][i]
        places = places[1]
        for place_id, score in zip(places.split(), static_scores*weights[j]):
            if dict.get(place_id) != None:
                dict[place_id] += score
            else:
                dict[place_id] = score   

    #print(dict)
    top3 = sorted(dict, key=dict.__getitem__, reverse=True)[:n_topx_out]
    #print(top3)
    #places_merged.append(str(top3[0]) + ' ' + str(top3[1]) + ' ' + str(top3[2]))
    top3 = [int(x) for x in top3]
    places_merged.append(str(top3).replace(',', '').replace('[', '').replace(']', ''))
    
    if i % 1000000 == 0:
        print(i)

duration = (datetime.datetime.now() - start_time).seconds//60
print('merge done, duration', duration)
 
sub = pd.DataFrame(np.column_stack((slices[0][:, 0], places_merged)), columns=['row_id', 'place_id'])
sub.to_csv('results_en_' + str(n_topx_out) + '.csv', index=False)   