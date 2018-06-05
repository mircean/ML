import numpy as np
import datetime

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn

import dataset
import models
import util

#TODO
#support hashing for longer n-grams
#if load model, synchronize random seed    

opt = {
    #'model': 'WordVecSum',
    #'model': 'WordLSTM1',
    'model': 'WordNGramCNN',
    #'model': 'CharCNN',
    'bow_tfidf': True,
    'bow_ngram_range': (1,2),
    'bow_min_df': 5,

    #'glove_file': r'C:\Users\mircea-mac\OneDrive\Glove\glove.840B.300d.txt',
    #'glove_file': r'C:\Users\mirce\OneDrive\Glove\glove.840B.300d.txt',
    #'glove_dim': 300,
    'glove_file': r'C:\Users\mirce\OneDrive\Glove\glove.6B.50d.txt',
    'glove_dim': 50,

    'vocabulary_padding_id': 0,
    'vocabulary_unknown_id': 1,

    'cuda': True,
    #'batch_size': 256
    #128 for CharCNN
    'batch_size': 128
    }

class Trainer:
    def __init__(self, dataset, num_classes=None):
        self.dataset = dataset
        self.num_classes = num_classes
        
    def train(self):
        raise NotImplementedError

class BOWTrainer(Trainer):
    def __init__(self, dataset):
        super(BOWTrainer, self).__init__(dataset, num_classes=None)    

        print(datetime.datetime.now(), 'Building vocabulary')
        self.vocabulary = util.build_vocabulary(self.dataset.df_train['words'].values)

        print(datetime.datetime.now(), 'Preparing data')
        #ids_string is a hack to make *Vectorizer work with a given dictionary so we have more flexibility when building the dictionary
        self.dataset.df_train['ids'] = self.dataset.df_train['words'].apply(lambda x: util.words2ids(x, self.vocabulary))
        self.dataset.df_train['ids_string'] = self.dataset.df_train['ids'].apply(lambda x: ' '.join([str(word).zfill(4) for word in x]))
        self.dataset.df_test['ids'] = self.dataset.df_test['words'].apply(lambda x: util.words2ids(x, self.vocabulary))
        self.dataset.df_test['ids_string'] = self.dataset.df_test['ids'].apply(lambda x: ' '.join([str(word).zfill(4) for word in x]))
        
    def train(self):
        X_train = self.dataset.df_train['ids_string'].values
        y_train = self.dataset.df_train['label'].values
        X_test = self.dataset.df_test['ids_string'].values
        y_test = self.dataset.df_test['label'].values

        print(datetime.datetime.now(), 'Vectorizing')
        if opt['bow_tfidf'] == False:
            self.cv = CountVectorizer(ngram_range=opt['bow_ngram_range'], min_df=opt['bow_min_df'])
            X_train = self.cv.fit_transform(X_train)
            X_test = self.cv.transform(X_test)
        else:
            self.tfidf = TfidfVectorizer(ngram_range=opt['bow_ngram_range'], min_df=opt['bow_min_df'])
            X_train = self.tfidf.fit_transform(X_train)
            X_test = self.tfidf.transform(X_test)

        #TODO: use sparse.vstack
        X_train = np.concatenate((X_train.todense(), self.dataset.df_train[self.dataset.features].values), axis=1)
        X_test = np.concatenate((X_test.todense(), self.dataset.df_test[self.dataset.features].values), axis=1)

        print(datetime.datetime.now(), 'Traing')
        self.lr = LogisticRegression()
        self.lr.fit(X_train, y_train)

        y_predict = self.lr.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_predict)
        y_predict = self.lr.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_predict)
        print(datetime.datetime.now(), (accuracy_train, accuracy_test))

    def eval(self):
        X_test = self.dataset.df_test['ids_string'].values
        y_test = self.dataset.df_test['label'].values

        print(datetime.datetime.now(), 'Vectorizing')
        if opt['bow_tfidf'] == False:
            X_test = self.cv.transform(X_test)
        else:
            X_test = self.tfidf.transform(X_test)

        #TODO: use sparse.vstack
        X_test = np.concatenate((X_test.todense(), self.dataset.df_test[self.dataset.features].values), axis=1)

        y_predict = self.lr.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_predict)
        print(datetime.datetime.now(), accuracy_test)
        print(confusion_matrix(y_test, y_predict))

class NNTrainer(Trainer):
    def __init__(self, dataset, num_classes):
        super(NNTrainer, self).__init__(dataset, num_classes)    
    
        if opt['model'] in ['WordVecSum', 'WordLSTM1', 'WordNGramCNN']:
            if self.dataset.status != 'WordLevel':
                self.dataset.prepare_wordlevel(opt)
            self.embeddings = self.dataset.embeddings

        if opt['model'] == 'CharCNN':
            if self.dataset.status != 'CharLevel':
                self.dataset.prepare_charlevel()

        self.create_model()

    def reset(self):
        self.create_model()
        
    def create_model(self):
        print(datetime.datetime.now(), 'Creating model')
        if opt['model'] == 'WordVecSum':
            self.model = models.WordVecSum(self.embeddings, self.num_classes)
        elif opt['model'] == 'WordLSTM1':
            self.model = models.WordLSTM1(self.embeddings, self.num_classes)
        elif opt['model'] == 'WordNGramCNN':
            self.model = models.WordNGramCNN(self.embeddings, self.num_classes)
        elif opt['model'] == 'CharCNN':
            self.model = models.CharCNN(self.dataset.char2ids, self.num_classes)
        else: raise NotImplementedError
        
        if opt['cuda']:
            self.model.cuda()
        
        self.loss = torch.nn.BCELoss() if self.num_classes == 2 else torch.nn.CrossEntropyLoss()
        #self.loss = torch.nn.CrossEntropyLoss(weights)

        if opt['model'] in ['WordVecSum', 'WordNGramCNN', 'CharCNN']:
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-2)
        elif opt['model'] == 'WordLSTM1':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        else: raise NotImplementedError

        self.epoch = 0        

    def create_batch(self, X, Y, indices):
        m_batch = len(indices)
        X_batch = X[indices]
        Y_batch = Y[indices]
        
        X_batch2 = torch.LongTensor(m_batch, max([len(doc) for doc in X_batch])).fill_(0)
        X_mask_batch = torch.LongTensor(m_batch).fill_(0)
        for index, doc in enumerate(X_batch):
            X_batch2[index, :len(doc)] = torch.LongTensor(doc)
            X_mask_batch[index] = len(doc)

        X_batch = X_batch2
        X_mask_batch = X_mask_batch.unsqueeze(1)
        #workaround for div
        X_mask_batch = torch.FloatTensor(X_mask_batch.numpy().astype(float))

        if self.num_classes == 2:
            #Float for BCELoss
            Y_batch = torch.FloatTensor(Y_batch)
        else:
            Y_batch = torch.LongTensor(Y_batch)
        
        if opt['cuda']:
            X_batch = X_batch.cuda()
            X_mask_batch = X_mask_batch.cuda()
            Y_batch = Y_batch.cuda()
            
        return X_batch, X_mask_batch, Y_batch
        
    def create_batch_char(self, X, Y, indices):
        m_batch = len(indices)
        X_batch = dataset.documents2char(X[indices], self.dataset.char2ids)
        X_batch = torch.FloatTensor(X_batch)
        
        Y_batch = Y[indices]
        if self.num_classes == 2:
            #Float for BCELoss
            Y_batch = torch.FloatTensor(Y_batch)
        else:
            Y_batch = torch.LongTensor(Y_batch)
        
        if opt['cuda']:
            X_batch = X_batch.cuda()
            Y_batch = Y_batch.cuda()
            
        return X_batch, None, Y_batch

    def calculate_accuracy(self, Y_true, Y_predict):
        if self.num_classes == 2:
            Y_predict = (Y_predict >= 0.5).float()
        else:
            _, Y_predict = torch.max(Y_predict, 1)

        correct = (Y_predict == Y_true).sum()
        correct = correct.cpu().data.numpy()
        #can be removed. type is int in the official pytorch
        #if 'numpy.ndarray' in str(type(correct)):
        #    correct = correct[0]
        accuracy = correct/Y_true.size(0)
        return accuracy, Y_predict

    def test(self):
        if opt['model'] == 'CharCNN':
            X_train = self.dataset.df_train['text_parsed'].values
            X_test = self.dataset.df_test['text_parsed'].values
        else:
            X_train = self.dataset.df_train['ids'].values
            X_test = self.dataset.df_test['ids'].values

        Y_train = self.dataset.df_train['label'].values
        Y_test = self.dataset.df_test['label'].values        

        m_train = len(X_train)
        permutation = torch.randperm(m_train)

        accuracies = []
        for start_idx in range(0, m_train, opt['batch_size']):
            indices = permutation[start_idx:start_idx + opt['batch_size']]

            if opt['model'] == 'CharCNN':
                X_train_batch, X_train_mask_batch, Y_train_batch = self.create_batch_char(X_train, Y_train, indices)
            else:
                X_train_batch, X_train_mask_batch, Y_train_batch = self.create_batch(X_train, Y_train, indices)
            Y_predict = self.model(X_train_batch, X_train_mask_batch)
            loss = self.loss(Y_predict, Y_train_batch)

            accuracy, _ = self.calculate_accuracy(Y_train_batch, Y_predict)
            accuracies.append(accuracy)
            print(loss.cpu().data.numpy(), accuracy)

            del X_train_batch, X_train_mask_batch, Y_train_batch, Y_predict

        print(sum(accuracies)/len(accuracies))

    def train(self, epochs=1000, epochs_eval=0, verbose=False):
        print(datetime.datetime.now(), 'Training')
        if opt['model'] == 'CharCNN':
            X_train = self.dataset.df_train['text_parsed'].values
            X_test = self.dataset.df_test['text_parsed'].values
        else:
            X_train = self.dataset.df_train['ids'].values
            X_test = self.dataset.df_test['ids'].values

        Y_train = self.dataset.df_train['label'].values
        Y_test = self.dataset.df_test['label'].values  

        m_train = len(X_train)
        m_test = len(X_test)
        
        accuracy_test_best = 0
        
        for epoch_local in range(epochs):
            #Forward pass
            self.model.train()

            #shuffle data
            permutation = torch.randperm(m_train)
            
            losses = []
            accuracies = []
            batch_weights = []
            for start_idx in range(0, m_train, opt['batch_size']):
                if verbose: print('Train epoch/index/total', epoch_local, start_idx, m_train)
                indices = permutation[start_idx:start_idx + opt['batch_size']]
                
                if opt['model'] == 'CharCNN':
                    X_train_batch, X_train_mask_batch, Y_train_batch = self.create_batch_char(X_train, Y_train, indices)
                else:
                    X_train_batch, X_train_mask_batch, Y_train_batch = self.create_batch(X_train, Y_train, indices)

                Y_predict = self.model(X_train_batch, X_train_mask_batch)
                loss = self.loss(Y_predict, Y_train_batch)

                accuracy, _ = self.calculate_accuracy(Y_train_batch, Y_predict)
                losses.append(loss.cpu().data.numpy())
                accuracies.append(accuracy)
                batch_weights.append(len(indices))

                #Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                del X_train_batch, X_train_mask_batch, Y_train_batch, Y_predict

            if epochs_eval != 0 and self.epoch % epochs_eval == 0:
                #Calculate train and test accuracy
                loss = np.average(losses, weights=batch_weights)
                accuracy_train = np.average(accuracies, weights=batch_weights)
                accuracy_test, _ = self.eval_internal(X_test, Y_test, verbose)
                #can be removed. type is int in the official pytorch
                #if 'numpy.ndarray' in str(type(loss)):
                #    loss = loss[0]

                print("epoch {0:06d} loss {1:.4f} train acc {2:.4f} test acc {3:.4f}".format(self.epoch, loss, accuracy_train, accuracy_test))
                if accuracy_test > accuracy_test_best:
                    accuracy_test_best = accuracy_test
                    #torch.save(model, 'models/model' + str(epoch))

            self.epoch += 1        

    def eval(self):
        if opt['model'] == 'CharCNN':
            X_test = self.dataset.df_test['text_parsed'].values
        else:
            X_test = self.dataset.df_test['ids'].values

        Y_test = self.dataset.df_test['label'].values  

        accuracy, confusion = self.eval_internal(X_test, Y_test)
        print(datetime.datetime.now(), accuracy)
        print(confusion)
    
    def eval_internal(self, X_test, Y_test, verbose=False):
        self.model.eval()

        m_test = len(X_test)

        accuracies = []
        losses = []
        batch_weights = []
        
        Y_test_all = np.array([])
        Y_predict_all = np.array([])
        
        for start_idx in range(0, m_test, opt['batch_size']):
            if verbose: print('Eval epoch/index/total', epoch_local, start_idx, m_test)
            indices = [start_idx + i for i in range(opt['batch_size']) if start_idx + i < m_test ]
            if opt['model'] == 'CharCNN':
                X_test_batch, X_test_mask_batch, Y_test_batch = self.create_batch_char(X_test, Y_test, indices)
            else:
                X_test_batch, X_test_mask_batch, Y_test_batch = self.create_batch(X_test, Y_test, indices)

            Y_predict = self.model(X_test_batch, X_test_mask_batch) 
            accuracy, Y_predict = self.calculate_accuracy(Y_test_batch, Y_predict)
            accuracies.append(accuracy)
            batch_weights.append(len(indices))
            
            Y_test_all = np.concatenate((Y_test_all, Y_test_batch.cpu().data.numpy()))
            Y_predict_all = np.concatenate((Y_predict_all, Y_predict.cpu().detach().numpy()))
            del X_test_batch, X_test_mask_batch, Y_test_batch, Y_predict

        accuracy = np.average(accuracies, weights=batch_weights)
        return accuracy, confusion_matrix(Y_test_all, Y_predict_all)
