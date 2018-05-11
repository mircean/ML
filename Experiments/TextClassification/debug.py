import numpy as np
import pandas as pd
import datetime

import torch

import dataset
import models

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if models.opt['cuda']:
    torch.cuda.manual_seed_all(seed)
#TODO: if load model, synchronize random seed 

from os import listdir
from os.path import isfile, join
#imdb_folder = r'C:\Users\mirce\OneDrive\Datasets\aclImdb'
imdb_folder = r'C:\Users\mircea-mac\OneDrive\Datasets\aclImdb'

def read_reviews(folder):
    reviews = []
    for file in listdir(folder + r'\pos'):
        with open(folder + r'\pos' + '\\' + file, encoding='utf-8') as f:
            review = f.read()
            reviews.append([review, 1])
        #if len(reviews) >= 20: break
    for file in listdir(folder + r'\neg'):
        with open(folder + r'\neg' + '\\' + file, encoding='utf-8') as f:
            review = f.read()
            reviews.append([review, 0])
        #if len(reviews) >= 20: break
    return pd.DataFrame(reviews, columns=['text', 'label'])
 
preprocess = False
save = False
if preprocess:
    df_train = read_reviews(imdb_folder + r'\train')
    df_test = read_reviews(imdb_folder + r'\test')
    data = dataset.Dataset(None, df_train, df_test)
    data.preprocess()
    data.df_train['words'] = data.df_train['words'].apply(lambda x: ' '.join(x))
    data.df_test['words'] = data.df_test['words'].apply(lambda x: ' '.join(x))
    if save:
        data.df_train.to_csv(imdb_folder + r'\train.csv', index=False)
        data.df_test.to_csv(imdb_folder + r'\test.csv', index=False)
else:
    df_train = pd.read_csv(imdb_folder + r'\train.csv')
    df_train['words'] = df_train['words'].apply(lambda x: x.split(' '))
    df_test = pd.read_csv(imdb_folder + r'\test.csv')
    df_test['words'] = df_test['words'].apply(lambda x: x.split(' '))
    data = dataset.Dataset(None, df_train, df_test)


trainer = models.NNTrainer(data, num_classes=2)
trainer.train(epochs=10, epochs_eval=1, verbose=True)

