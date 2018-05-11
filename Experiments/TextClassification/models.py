import numpy as np
import datetime

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable

import dataset

#TODO
#support hashing for longer n-grams

opt = {
    'model': 'LSTM1',
    #'model': 'NGramCNN',
    'bow_tfidf': True,
    'bow_ngram_range': (1,2),
    'bow_min_df': 5,

        'glove_file': r'C:\Users\mircea-mac\OneDrive\Glove\glove.840B.300d.txt',
        'glove_dim': 300,

        'vocabulary_padding_id': 0,
        'vocabulary_unknown_id': 1,

        'cuda': False,
        'batch_size': 256
        }

def build_vocabulary(documents, glove_vocabulary=None):
    words = {'<PAD>': {'count': 0, 'id': 0}, '<UNK>': {'count': 0, 'id': 1}}
    ids = ['<PAD>', '<UNK>']
    
    for document in documents:
        for word in document:
            if word not in words:
                if glove_vocabulary == None:
                    words[word] = {'count': 0, 'id': len(words)}
                    ids.append(word)
                else:
                    if word in glove_vocabulary:
                        words[word] = {'count': 0, 'id': len(words)}
                        ids.append(word)
            if word in words:
                words[word]['count'] = words[word]['count'] + 1

    return {'words': words, 'ids': ids}

def words2ids(words, vocabulary, unknown='ignore'):
    #word not in vocabulary if
    #a. test set
    #b. word not in glove
    #unknown can be
    #'ignore'
    #'unknown_id'
    #'fail'
    if unknown == 'fail':
        return [vocabulary['words'][word]['id'] for word in words]
    elif unknown == 'unkwown_id':
        return [vocabulary['words'][word]['id'] if word in vocabulary['words'] else vocabulary_unknown_id for word in words]
    else:
        return [vocabulary['words'][word]['id'] for word in words if word in vocabulary['words']]
        
def load_glove_vocabulary():
    vocabulary = set()
    with open(opt['glove_file'], encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = dataset.normalize_text(''.join(elems[0:-opt['glove_dim']]))
            vocabulary.add(token)
    return vocabulary

def build_embeddings(vocabulary):
    vocabulary_size = len(vocabulary['words'])
    embeddings = np.random.uniform(-1, 1, (vocabulary_size, opt['glove_dim']))
    embeddings[0] = 0 # <PAD> should be all 0 (using broadcast)

    with open(opt['glove_file'], encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = dataset.normalize_text(''.join(elems[0:-opt['glove_dim']]))
            if token in vocabulary['words']:
                embeddings[vocabulary['words'][token]['id']] = [float(v) for v in elems[-opt['glove_dim']:]]
    return embeddings

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
        vocabulary = build_vocabulary(self.dataset.df_train['words'].values)

        print(datetime.datetime.now(), 'Preparing data')
        self.dataset.df_train['ids'] = self.dataset.df_train['words'].apply(lambda x: words2ids(x, vocabulary))
        self.dataset.df_train['ids_string'] = self.dataset.df_train['ids'].apply(lambda x: ' '.join([str(word).zfill(4) for word in x]))
        self.dataset.df_test['ids'] = self.dataset.df_test['words'].apply(lambda x: words2ids(x, vocabulary))
        self.dataset.df_test['ids_string'] = self.dataset.df_test['ids'].apply(lambda x: ' '.join([str(word).zfill(4) for word in x]))
        
    def train(self):
        X_train = self.dataset.df_train['ids_string'].values
        y_train = self.dataset.df_train['label'].values
        X_test = self.dataset.df_test['ids_string'].values
        y_test = self.dataset.df_test['label'].values

        print(datetime.datetime.now(), 'Vectorizing')
        if opt['bow_tfidf'] == False:
            cv = CountVectorizer(ngram_range=opt['bow_ngram_range'], min_df=opt['bow_min_df'])
            X_train = cv.fit_transform(X_train)
            X_test = cv.transform(X_test)
        else:
            tfidf = TfidfVectorizer(ngram_range=opt['bow_ngram_range'], min_df=opt['bow_min_df'])
            X_train = tfidf.fit_transform(X_train)
            X_test = tfidf.transform(X_test)

        print(datetime.datetime.now(), 'Traing')
        lr = LogisticRegression()
        lr.fit(X_train, y_train)

        y_predict = lr.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_predict)
        y_predict = lr.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_predict)
        print(datetime.datetime.now(), (accuracy_train, accuracy_test))

class NNTrainer(Trainer):
    def __init__(self, dataset, num_classes):
        super(NNTrainer, self).__init__(dataset, num_classes)    
    
        #if opt['model'] in ['WordVecSum', 'NGramCNN']:

        print(datetime.datetime.now(), 'Loading Glove')
        glove_vocabulary = load_glove_vocabulary()
        print(datetime.datetime.now(), 'Building vocabulary')
        vocabulary = build_vocabulary(self.dataset.df_train['words'].values, glove_vocabulary)
        
        print(datetime.datetime.now(), 'Preparing data')
        self.dataset.df_train['ids'] = self.dataset.df_train['words'].apply(lambda x: words2ids(x, vocabulary))
        self.dataset.df_test['ids'] = self.dataset.df_test['words'].apply(lambda x: words2ids(x, vocabulary))

        print(datetime.datetime.now(), 'Buidling embeddings')
        self.embeddings = build_embeddings(vocabulary)
    
        self.create_model()

    def reset(self):
        self.create_model()
        
    def create_model(self):
        print(datetime.datetime.now(), 'Creating model')
        if opt['model'] == 'WordVecSum':
            self.model = WordVecSum(self.embeddings, self.num_classes)
        elif opt['model'] == 'NGramCNN':
            self.model = NGramCNN(self.embeddings, self.num_classes)
        elif opt['model'] == 'LSTM1':
            self.model = LSTM1(self.embeddings, self.num_classes)
        else: raise NotImplementedError
        
        if opt['cuda']:
            self.model.cuda()
        
        self.loss = torch.nn.BCELoss() if self.num_classes == 2 else torch.nn.CrossEntropyLoss()
        #self.loss = torch.nn.CrossEntropyLoss(weights)

        if opt['model'] in ['WordVecSum', 'NGramCNN']:
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-2)
        elif opt['model'] == 'LSTM1':
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
            Y_batch = Variable(torch.FloatTensor(Y_batch), requires_grad=False)
        else:
            Y_batch = Variable(torch.LongTensor(Y_batch), requires_grad=False)
        
        if opt['cuda']:
            X_batch = X_batch.cuda()
            X_mask_batch = X_mask_batch.cuda()
            Y_batch = Y_batch.cuda()
            
        return X_batch, X_mask_batch, Y_batch
        
    def calculate_accuracy(self, Y_true, Y_predict):
        if self.num_classes == 2:
            Y_predict = Y_predict >= 0.5
        else:
            _, Y_predict = torch.max(Y_predict, 1)

        correct = (Y_predict.float() == Y_true).sum()
        correct = correct.cpu().data.numpy()
        #can be removed. type is int in the official pytorch
        if 'numpy.ndarray' in str(type(correct)):
            correct = correct[0]
        accuracy = correct/Y_true.size(0)
        return accuracy


    def test(self):
        X_train = self.dataset.df_train['ids'].values
        Y_train = self.dataset.df_train['label'].values
        X_test = self.dataset.df_test['ids'].values
        Y_test = self.dataset.df_test['label'].values        

        m_train = len(X_train)
        permutation = torch.randperm(m_train)

        accuracies = []
        for start_idx in range(0, m_train, opt['batch_size']):
            indices = permutation[start_idx:start_idx + opt['batch_size']]

            X_train_batch, X_train_mask_batch, Y_train_batch = self.create_batch(X_train, Y_train, indices)
            Y_predict = self.model(X_train_batch, X_train_mask_batch)
            loss = self.loss(Y_predict, Y_train_batch)

            accuracy = self.calculate_accuracy(Y_train_batch, Y_predict)
            accuracies.append(accuracy)
            print(loss.cpu().data.numpy(), accuracy)

        print(sum(accuracies)/len(accuracies))

    def train(self, epochs=1000, epochs_eval=0, verbose=False):
        X_train = self.dataset.df_train['ids'].values
        Y_train = self.dataset.df_train['label'].values
        X_test = self.dataset.df_test['ids'].values
        Y_test = self.dataset.df_test['label'].values  

        m_train = len(X_train)
        m_test = len(X_test)
        
        accuracy_test_best = 0
        
        for epoch_local in range(epochs):
            #Forward pass
            self.model.train()

            #shuffle data
            permutation = torch.randperm(m_train)
            
            accuracies = []
            accuracies_weights = []
            for start_idx in range(0, m_train, opt['batch_size']):
                if verbose: print('Train epoch/index/total', epoch_local, start_idx, m_train)
                indices = permutation[start_idx:start_idx + opt['batch_size']]
                
                X_train_batch, X_train_mask_batch, Y_train_batch = self.create_batch(X_train, Y_train, indices)
                Y_predict = self.model(X_train_batch, X_train_mask_batch)
                loss = self.loss(Y_predict, Y_train_batch)

                accuracy = self.calculate_accuracy(Y_train_batch, Y_predict)
                accuracies.append(accuracy)
                accuracies_weights.append(len(indices))

                #Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epochs_eval != 0 and self.epoch % epochs_eval == 0:
                #Calculate train and test accuracy
                accuracy_train = np.average(accuracies, weights=accuracies_weights)

                self.model.eval()

                accuracies = []
                accuracies_weights = []
                
                for start_idx in range(0, m_test, opt['batch_size']):
                    if verbose: print('Eval epoch/index/total', epoch_local, start_idx, m_test)
                    indices = [start_idx + i for i in range(opt['batch_size']) if start_idx + i < m_test ]
                    X_test_batch, X_test_mask_batch, Y_test_batch = self.create_batch(X_test, Y_test, indices)
                    Y_predict = self.model(X_test_batch, X_test_mask_batch) 
                    accuracy = self.calculate_accuracy(Y_test_batch, Y_predict)
                    accuracies.append(accuracy)
                    accuracies_weights.append(len(indices))

                accuracy_test = np.average(accuracies, weights=accuracies_weights)
                
                loss = loss.cpu().data.numpy()
                #can be removed. type is int in the official pytorch
                if 'numpy.ndarray' in str(type(loss)):
                    loss = loss[0]

                print("epoch {0:06d} loss {1:.4f} train acc {2:.4f} test acc {3:.4f}".format(self.epoch, loss, accuracy_train, accuracy_test))
                if accuracy_test > accuracy_test_best:
                    accuracy_test_best = accuracy_test
                    #torch.save(model, 'models/model' + str(epoch))

            self.epoch += 1        

class WordVecSum(nn.Module):
    def __init__(self, embeddings, num_classes=2):
        super(WordVecSum, self).__init__()    
    
        self.num_classes = num_classes
        
        #embedding layer
        self.embedding_dim = embeddings.shape[1]
        self.embedding = nn.Embedding(embeddings.shape[0],  #vocab size
                                      self.embedding_dim,   #embedding_dim
                                      padding_idx=0)
        self.embedding.weight.data = torch.Tensor(embeddings)
        #do not backprop into embeddings
        for p in self.embedding.parameters():
            p.requires_grad = False
        
        #liniar layerm + sigmoid/softmax
        if self.num_classes == 2:
            self.linear = nn.Linear(self.embedding_dim, 1)    
            self.sigmoid = nn.Sigmoid()
        else:
            self.linear = nn.Linear(self.embedding_dim, num_classes)
     
    def forward(self, X, X_mask):
        #X: [m, Tx] m = batch size, Tx = word count
        #print(X.size(), type(X))
        m = X.size()[0]
        Tx = X.size()[1]
        
        X = self.embedding(X)
        #X: [m, Tx, embedding_dim] m = batch size, Tx = word count
        #print(X.size(), type(X.data))
        assert X.size() == torch.Size([m, Tx, self.embedding_dim])
                
        #average words in doc. use mask so we average only words not padding
        X = torch.sum(X, 1)
        X = Variable(torch.div(X.data, X_mask))
        #X: [m, emb_dim]
        #print(X.size(), type(X.data))
        assert X.size() == torch.Size([m, self.embedding_dim])
        
        X = self.linear(X)
        #X: [m, 1]
        #print(X.size(), type(X))
        if self.num_classes == 2:
            assert X.size() == torch.Size([m, 1])
        else:
            assert X.size() == torch.Size([m, self.num_classes])
            
        if self.num_classes == 2:
            X = torch.squeeze(X)
            X = self.sigmoid(X)
            #X: [m]
            #print(X.size(), type(X))
            assert X.size() == torch.Size([m])
            return X
        else:
            return F.softmax(X)
            
class LSTM1(nn.Module):
    def __init__(self, embeddings, num_classes=2):
        super(LSTM1, self).__init__()    
    
        self.num_classes = num_classes
        
        #embedding layer
        self.embedding_dim = embeddings.shape[1]
        self.embedding = nn.Embedding(embeddings.shape[0],  #vocab size
                                      self.embedding_dim,   #embedding_dim
                                      padding_idx=0)
        self.embedding.weight.data = torch.Tensor(embeddings)
        #do not backprop into embeddings
        for p in self.embedding.parameters():
            p.requires_grad = False

        #LSTM1, hidden_size = 128
        #TODO: try bidirectional=True
        self.LSTM1_hidden_size = 128
        self.LSTM1 = nn.LSTM(self.embedding_dim, self.LSTM1_hidden_size, bidirectional = False)
        #self.LSTM1 = nn.LSTM(self.embedding_dim, self.LSTM1_hidden_size, bidirectional = True)
        
        #dropout
        self.dropout = nn.Dropout()
     
        #LSTM, hidden_size = 128
        #TODO: try bidirectional=True
        self.LSTM2_hidden_size = 128
        self.LSTM2 = nn.LSTM(self.LSTM1_hidden_size, self.LSTM2_hidden_size, bidirectional = False)
        #self.LSTM2 = nn.LSTM(2*self.LSTM1_hidden_size, self.LSTM2_hidden_size, bidirectional = False)
        
        #liniar layerm + sigmoid/softmax
        if self.num_classes == 2:
            self.linear = nn.Linear(self.LSTM2_hidden_size, 1)    
            self.sigmoid = nn.Sigmoid()
        else:
            self.linear = nn.Linear(self.LSTM2_hidden_size, num_classes)

    def forward(self, X, X_mask, verbose=False):
        #X: [m, Tx] m = batch size, Tx = word count
        if verbose: print(X.size(), type(X))
        m = X.size()[0]
        Tx = X.size()[1]

        #embedding layer
        X = self.embedding(X)
        #X: [m, Tx, embedding_dim] 
        if verbose: print(X.size(), type(X.data))
        assert X.size() == torch.Size([m, Tx, self.embedding_dim])
           
        #LSTM1
        # Transpose batch and sequence dims
        X = X.transpose(0, 1)
        X, _ = self.LSTM1(X)
        # Transpose back
        X = X.transpose(0, 1)
        #X: [m, Tx, LSTM1_hidden_size] 
        if verbose: print(X.size(), type(X.data))
        assert X.size() == torch.Size([m, Tx, self.LSTM1_hidden_size])
        #assert X.size() == torch.Size([m, Tx, 2*self.LSTM1_hidden_size])
        
        #dropout
        X = self.dropout(X)

        #LSTM2, reduce dimension
        # Transpose batch and sequence dims
        X = X.transpose(0, 1)
        _, X = self.LSTM2(X)
        X = X[0]
        # Transpose back
        X = X.transpose(0, 1)
        X = torch.squeeze(X)
        #X: [m, LSTM2_hidden_size] 
        if verbose: print(X.size(), type(X.data))
        assert X.size() == torch.Size([m, self.LSTM2_hidden_size])
        
        #dropout
        X = self.dropout(X)

        #linear
        X = self.linear(X)
        #X: [m, 1]
        if verbose: print(X.size(), type(X))
        if self.num_classes == 2:
            assert X.size() == torch.Size([m, 1])
        else:
            assert X.size() == torch.Size([m, self.num_classes])
        
        if self.num_classes == 2:
            X = torch.squeeze(X)
            X = self.sigmoid(X)
            #X: [m]
            if verbose: print(X.size(), type(X))
            assert X.size() == torch.Size([m])
            return X
        else:
            return F.softmax(X)
    
class NGramCNN(nn.Module):
    def __init__(self, embeddings, num_classes=2):
        super(NGramCNN, self).__init__()    
    
        self.num_classes = num_classes
        
        #embedding layer
        self.embedding_dim = embeddings.shape[1]
        self.embedding = nn.Embedding(embeddings.shape[0],  #vocab size
                                      self.embedding_dim,   #embedding_dim
                                      padding_idx=0)
        self.embedding.weight.data = torch.Tensor(embeddings)
        #do not backprop into embeddings
        for p in self.embedding.parameters():
            p.requires_grad = False

        #conv layer
        self.conv = nn.Conv1d(self.embedding_dim, 256, kernel_size=3, stride=1, padding=1)        

        #max pool layer
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)

        #liniar layerm + sigmoid/softmax
        if self.num_classes == 2:
            self.linear = nn.Linear(256, 1)    
            self.sigmoid = nn.Sigmoid()
        else:
            self.linear = nn.Linear(256, num_classes)

    def forward(self, X, X_mask):
        #X: [m, Tx] m = batch size, Tx = word count
        #print(X.size(), type(X))
        m = X.size()[0]
        Tx = X.size()[1]
        
        #embedding layer
        X = self.embedding(X)
        #X: [m, Tx, embedding_dim] m = batch size, Tx = word count
        #print(X.size(), type(X.data))
        assert X.size() == torch.Size([m, Tx, self.embedding_dim])
        
        #conv layer
        #transpose
        X = torch.transpose(X, 1, 2)
        #print(X.size(), type(X.data))

        X = self.conv(X)
        #print(X.size(), type(X.data))

        #transpose back
        X = torch.transpose(X, 1, 2)
        #print(X.size(), type(X.data))

        assert X.size() == torch.Size([m, Tx, 256])

        #maxpool layer
        #transpose
        X = torch.transpose(X, 1, 2)
        X = self.maxpool(X)
        #print(X.size(), type(X.data))
        #remove dimension
        X = X.squeeze()
        #print(X.size(), type(X.data))
        assert X.size() == torch.Size([m, 256])

        #linear 
        X = self.linear(X)
        #X: [m, 1]
        #print(X.size(), type(X))
        if self.num_classes == 2:
            assert X.size() == torch.Size([m, 1])
        else:
            assert X.size() == torch.Size([m, self.num_classes])
            
        if self.num_classes == 2:
            X = torch.squeeze(X)
            X = self.sigmoid(X)
            #X: [m]
            #print(X.size(), type(X))
            assert X.size() == torch.Size([m])
            return X
        else:
            return F.softmax(X)

