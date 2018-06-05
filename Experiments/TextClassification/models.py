import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable

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
            
class WordLSTM1(nn.Module):
    def __init__(self, embeddings, num_classes=2):
        super(WordLSTM1, self).__init__()    
    
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
    
class WordNGramCNN(nn.Module):
    def __init__(self, embeddings, num_classes=2):
        super(WordNGramCNN, self).__init__()    
    
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

class CharCNN(nn.Module):
    def __init__(self, char2ids, num_classes=2):
        super(CharCNN, self).__init__()    
    
        self.char2ids = char2ids
        self.num_classes = num_classes
        
        self.num_filters = 1024

        modules = []
        modules.append(nn.Conv1d(1, self.num_filters, (len(char2ids), 7)))
        modules.append(nn.MaxPool1d(kernel_size=2))
        modules.append(nn.Conv1d(1, self.num_filters, (self.num_filters, 7)))
        modules.append(nn.MaxPool1d(kernel_size=2))
        modules.append(nn.Conv1d(1, self.num_filters, (self.num_filters, 3)))
        modules.append(nn.MaxPool1d(kernel_size=2, padding=1))
        modules.append(nn.Conv1d(1, self.num_filters, (self.num_filters, 3)))
        modules.append(nn.MaxPool1d(kernel_size=2, padding=0))
        modules.append(nn.Conv1d(1, self.num_filters, (self.num_filters, 3)))
        modules.append(nn.MaxPool1d(kernel_size=2, padding=1))
        modules.append(nn.Conv1d(1, self.num_filters, (self.num_filters, 3)))
        modules.append(nn.MaxPool1d(kernel_size=2, padding=0))

        self.convlayers = nn.ModuleList(modules)

        #liniar layerm + sigmoid/softmax
        if self.num_classes == 2:
            self.linear = nn.Linear(14*self.num_filters, 1)    
            self.sigmoid = nn.Sigmoid()
        else:
            self.linear = nn.Linear(14*self.num_filters, num_classes)

    def forward(self, X, X_mask):
        #X_mask not used

        #X: [m, V, Tx] m = batch size, V = vocabulary size, Tx = char count (max 1014)
        #print(X.size(), type(X))
        m = X.size()[0]
        V = X.size()[1]
        Tx = X.size()[2]
        
        #conv layer        
        for module in self.convlayers:
            #print(X.size(), type(X.data))
            if 'Conv1d' in str(type(module)): X = X.unsqueeze(1)
            X = module(X)
            if 'Conv1d' in str(type(module)): X = X.squeeze()

        #X: [m, num_filters, 14]
        #print(X.size(), type(X.data))
        assert X.size() == torch.Size([m, self.num_filters, 14])

        #linear 
        X = torch.reshape(X, (m, 14*self.num_filters))
        assert X.size() == torch.Size([m, self.num_filters*14])
        
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


