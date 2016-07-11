import numpy as np
import pandas as pd
import datetime
from os import listdir

from common import merge, merge2

n_topx = 7 #was 3        
print('start', datetime.datetime.now())

files = listdir('.')

slices = []
for f in files:
    slice = pd.read_csv(f)
    slices.append(slice.values)

print('files read', len(slices))

results = merge(slices)
#results = merge2(slices, submit = True, n_topx = n_topx)

sub = pd.DataFrame(results, columns=['row_id', 'place_id'])
sub.to_csv('results_' + str(n_topx) + '.csv', index=False)            

print('done', datetime.datetime.now())