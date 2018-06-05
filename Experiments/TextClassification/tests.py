import numpy as np
import pandas as pd
import datetime

import dataset
import trainer

def test1_emoji():
    df_train = pd.read_csv('emoji/train_emoji.csv', header=None, usecols=[0,1], names=['text', 'label'])
    df_test = pd.read_csv('emoji/tesss.csv', header=None, usecols=[0,1], names=['text', 'label'])

    data = dataset.Dataset(None, df_train, df_test)

    print('***BOWTrainer')
    trainer.opt['bow_ngram_range'] = (1, 1)
    trainer.opt['bow_min_df'] = 1
    mytrainer = trainer.BOWTrainer(data)
    mytrainer.train()
    mytrainer.eval()

    print('***NNTrainer:WordVecSum')
    trainer.opt['model'] = 'WordVecSum'
    mytrainer = trainer.NNTrainer(data, 2)
    mytrainer.train(epochs=10, epochs_eval=1)
    mytrainer.eval()

    print('***NNTrainer:WordLSTM1')
    trainer.opt['model'] = 'WordLSTM1'
    mytrainer = trainer.NNTrainer(data, 2)
    mytrainer.train(epochs=10, epochs_eval=1)
    mytrainer.eval()
    
    print('***NNTrainer:WordNGramCNN')
    trainer.opt['model'] = 'WordNGramCNN'
    mytrainer = trainer.NNTrainer(data, 2)
    mytrainer.train(epochs=10, epochs_eval=1)
    mytrainer.eval()
    
    print('***NNTrainer:CharCNN')
    trainer.opt['model'] = 'CharCNN'
    mytrainer = trainer.NNTrainer(data, 2)
    mytrainer.train(epochs=10, epochs_eval=1)
    mytrainer.eval()
    
test1_emoji()
