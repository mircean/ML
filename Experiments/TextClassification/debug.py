import numpy as np
import pandas as pd
import datetime

import sys
sys.path.append(r'c:\IntelOpenmp')

import torch

import dataset
import trainer
import models

test = 2
if test == 1:
    from os import listdir
    from os.path import isfile, join
    imdb_folder = r'C:\Users\mirce\OneDrive\Datasets\aclImdb'
    #imdb_folder = r'C:\Users\mircea-mac\OneDrive\Datasets\aclImdb'

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


    mytrainer = trainer.NNTrainer(data, num_classes=2)
    mytrainer.train(epochs=100, epochs_eval=1)
    
if test == 2:
    file = r'C:\Users\mirce\OneDrive\Code\UiPath\Invoice\dataset-synthetic.tsv'
    df = pd.read_csv(file, sep='\t')
    df=df.rename(columns = {'value':'text'})
    train = int(df.shape[0]*0.8)
    df_train = pd.DataFrame(df[:train])
    df_test = pd.DataFrame(df[train:])

    data = dataset.Dataset(None, df_train, df_test)
    data.preprocess()

    trainer.opt['bow_ngram_range'] = (1, 1)
    trainer.opt['bow_min_df'] = 1

    mytrainer = trainer.BOWTrainer(data)
    mytrainer.train()
