import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import importlib

import sklearn

#normalization
from sklearn import preprocessing

#oe hot encoder
from sklearn.preprocessing import OneHotEncoder

#train/test split
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

#metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

#DNN
import DNN

#keras
np.random.seed(0) # keras needs the random seed before the import
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import regularizers

class KerasCallback(keras.callbacks.Callback):
    def __init__(self, verbose):
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose != None and epoch % self.verbose == 0:
            print(epoch, logs['loss'], logs['acc'], logs['val_acc'])        

def titanic_prep(api='dnn'):
    test_size=0.2

    df_train = pd.read_csv(r'titanic\train.csv')
    df_train.index = df_train.PassengerId

    y = df_train.Survived.values
    df_train = df_train.drop(['Survived'], axis=1)

    #remove columns
    df_train = df_train.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)

    #fill n/a
    df_train['Age'] = df_train.Age.fillna(-1)
    df_train['Embarked'] = df_train.Embarked.fillna('X')

    #one hot encoding. OK to do it for all data, inlcuding df_predict
    df_train['Sex'] = (df_train.Sex == 'male').astype(int)

    df_dummy = pd.get_dummies(df_train.Embarked, prefix='Embarked')
    df_train = df_train.drop(['Embarked'], axis=1)
    df_train = pd.concat((df_train, df_dummy), axis=1)

    X = df_train.values
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=test_size)

    print(X_train.shape)
    print(y_train.shape)
    print(X_dev.shape)
    print(y_dev.shape)

    #normalization - mean, var from train set, apply to dev set and test set
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)   
    X_dev = scaler.transform(X_dev)

    X_train_k = X_train
    X_dev_k = X_dev
    Y_train_k = y_train.reshape(y_train.shape[0], 1)
    Y_dev_k = y_dev.reshape(y_dev.shape[0], 1)

    X_train_dnn = X_train.T
    X_dev_dnn = X_dev.T
    Y_train_dnn = y_train.reshape(1, y_train.shape[0])
    Y_dev_dnn = y_dev.reshape(1, y_dev.shape[0])

    if api == 'dnn':
        return X_train_dnn, X_dev_dnn, Y_train_dnn, Y_dev_dnn
    elif api == 'keras':
        return X_train_k, X_dev_k, Y_train_k, Y_dev_k
    else:
        raise ValueError('api')

def titanic_dnn(X_train, X_dev, Y_train, Y_dev, random_seed=None):
    #random seed for parameters init, batch shuffle
    if random_seed != None:
        np.random.seed(random_seed)

    dnn = DNN.DNN()
    dnn.add_input_layer(X_train.shape[0])
    dnn.add_layer(20, np.tanh)
    dnn.add_layer(10, np.tanh)
    dnn.add_layer(1, DNN.sigmoid)
    dnn.costfunction = DNN.binary_crossentropy
    dnn.init = 'Lecun'
    dnn.optimizer = 'Adam'
    dnn.compile()    
    
    learning_rate = 0.001
    epochs = 10000
    batch_size = X_train.shape[1]
    gradient_check = False
    verbose = None

    #eval_set = None
    eval_set = [(X_train, Y_train), (X_dev, Y_dev)]

    results = dnn.fit(X_train, Y_train, eval_set=eval_set, eval_metric='error', learning_rate=learning_rate, epochs=epochs, batch_size = batch_size, gradient_check=gradient_check, verbose=verbose)

    Y_predict = dnn.predict(X_train)
    Y_predict = Y_predict > 0.5
    accuracy = accuracy_score(np.reshape(Y_train, Y_train.shape[1]), np.reshape(Y_predict, Y_predict.shape[1]))
    print('Accuracy train', accuracy)

    Y_predict = dnn.predict(X_dev)
    Y_predict = Y_predict > 0.5
    accuracy = accuracy_score(np.reshape(Y_dev, Y_dev.shape[1]), np.reshape(Y_predict, Y_predict.shape[1]))
    print('Accuracy dev', accuracy)

def titanic_keras(parameters, X_train, X_dev, Y_train, Y_dev, random_seed=None):
    #random seed
    if random_seed != None:
        np.random.seed(random_seed)
    
    hidden_1, hidden_2, activation, epochs, learning_rate, method, momentum = parameters

    if activation == 'tanh':
        initializer = keras.initializers.lecun_normal()
    elif activation == 'relu':
        initializer = keras.initializers.he_normal()
    else:
        raise ValueError('activation')

    regularizer = None
    #regularizer = regularizers.l2(0.01)

    model = Sequential()
    model.add(Dense(hidden_1, input_dim=X_train.shape[1], kernel_initializer=initializer, activation=activation, kernel_regularizer=regularizer))
    model.add(Dense(hidden_2, kernel_initializer=initializer, activation=activation, kernel_regularizer=regularizer))
    model.add(Dense(1, kernel_initializer=initializer, activation='sigmoid', kernel_regularizer=regularizer))

    if method == 'GD':
        optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)
    elif method == 'RMSProp':
        optimizer = optimizers.RMSprop(lr=learning_rate)
    elif method == 'Adam':
        optimizer = optimizers.Adam(lr=learning_rate)
    else:
        raise ValueError('method')

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    batch_size = X_train.shape[0]
    initial_epoch = 0
    verbose = 0
    #verbose2 = 10
    verbose2 = None

    results = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=[KerasCallback(verbose2)], validation_data = (X_dev, Y_dev), verbose=verbose, initial_epoch=initial_epoch)
    Y_predict = model.predict(X_train)
    Y_predict = Y_predict > 0.5
    accuracy_train = accuracy_score(Y_train, Y_predict)
    Y_predict = model.predict(X_dev)
    Y_predict = Y_predict > 0.5
    accuracy_dev = accuracy_score(Y_dev, Y_predict)

    print(accuracy_train, accuracy_dev, hidden_1, hidden_2, activation, epochs, learning_rate, method, momentum)

def MNIST_prep(api='dnn'):
    test_size = 0.05

    print('loading data')
    df_train = pd.read_csv(r'MNIST\train.csv')
    y = df_train.label.values
    print(y.shape)
    df_train = df_train.drop(['label'], axis=1)
    df_train /= 255
    print(df_train.shape)

    X = df_train.values
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=test_size, random_state=0)

    print(X_train.shape)
    print(y_train.shape)
    print(X_dev.shape)
    print(y_dev.shape)

    X_train_k = X_train
    X_dev_k = X_dev

    enc = OneHotEncoder()
    Y_train_k = enc.fit_transform(y_train.reshape(-1, 1)).todense().A
    Y_dev_k = enc.transform(y_dev.reshape(-1, 1)).todense().A

    X_train_dnn = X_train_k.T
    X_dev_dnn = X_dev_k.T
    Y_train_dnn = Y_train_k.T
    Y_dev_dnn = Y_dev_k.T

    if api == 'dnn':
        return X_train_dnn, X_dev_dnn, Y_train_dnn, Y_dev_dnn
    elif api == 'keras':
        return X_train_k, X_dev_k, Y_train_k, Y_dev_k
    else:
        raise ValueError('api')


def MNIST_dnn(parameters, X_train, X_dev, Y_train, Y_dev, random_seed=None):
    #random seed for parameters init, batch shuffle
    if random_seed != None:
        np.random.seed(random_seed)

    hidden_1, hidden_2, activation, epochs, learning_rate, batch_size, method, momentum = parameters
    if activation == 'tanh':
        initializer = 'Lecun'
        activation = np.tanh
    elif activation == 'relu':
        initializer = 'He'
        activation = DNN.relu
    else:
        raise ValueError('activation')

    regularizer = None
    #regularizer = regularizers.l2(0.01)

    dnn = DNN.DNN()
    dnn.add_input_layer(X_train.shape[0])
    dnn.add_layer(hidden_1, activation)
    dnn.add_layer(hidden_2, activation)
    dnn.add_layer(10, DNN.softmax)
    dnn.costfunction = DNN.categorical_crossentropy
    dnn.init = initializer
    if method == 'GD':
        optimizer = ('GD', momentum)
    else:
        optimizer = method
    dnn.optimizer = optimizer
    dnn.compile()

    gradient_check=False
    num_parameters = 2500
    verbose = None
    
    results = dnn.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_dev, Y_dev)], eval_metric='error', learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, gradient_check=gradient_check, num_parameters=num_parameters, verbose=verbose)

    Y_predict = dnn.predict(X_train)
    accuracy_train = accuracy_score(Y_predict.argmax(axis=0), Y_train.argmax(axis=0))
    Y_predict = dnn.predict(X_dev)
    accuracy_dev = accuracy_score(Y_predict.argmax(axis=0), Y_dev.argmax(axis=0))

    print(accuracy_train, accuracy_dev, hidden_1, hidden_2, activation, epochs, learning_rate, batch_size, method, momentum)

def MNIST_keras(parameters, X_train, X_dev, Y_train, Y_dev, random_seed=None):
    #random seed for parameters init, batch shuffle
    if random_seed != None:
        np.random.seed(random_seed)

    hidden_1, hidden_2, activation, epochs, learning_rate, batch_size, method, momentum = parameters

    if activation == 'tanh':
        initializer = keras.initializers.lecun_normal()
    elif activation == 'relu':
        initializer = keras.initializers.he_normal()
    else:
        raise ValueError('activation')

    regularizer = None
    #regularizer = regularizers.l2(0.01)

    model = Sequential()
    model.add(Dense(hidden_1, input_dim=X_train.shape[1], kernel_initializer=initializer, activation=activation, kernel_regularizer=regularizer))
    model.add(Dense(hidden_2, kernel_initializer=initializer, activation=activation, kernel_regularizer=regularizer))
    model.add(Dense(10, kernel_initializer=initializer, activation='softmax', kernel_regularizer=regularizer))

    if method == 'GD':
        optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)
    elif method == 'RMSProp':
        optimizer = optimizers.RMSprop(lr=learning_rate)
    elif method == 'Adam':
        optimizer = optimizers.Adam(lr=learning_rate)
    else:
        raise ValueError('method')

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    initial_epoch = 0
    verbose = 0
    #verbose2 = 10
    verbose2 = None
    results = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=[KerasCallback(verbose2)], validation_data = (X_dev, Y_dev), verbose=verbose, initial_epoch=initial_epoch)
    Y_predict = model.predict(X_train)
    accuracy_train = accuracy_score(Y_train.argmax(axis=1), Y_predict.argmax(axis=1))
    Y_predict = model.predict(X_dev)
    accuracy_dev = accuracy_score(Y_dev.argmax(axis=1), Y_predict.argmax(axis=1))

    print(accuracy_train, accuracy_dev, hidden_1, hidden_2, activation, epochs, learning_rate, batch_size, method, momentum)

def zillow_prep(api='dnn'):
    df1 = pd.read_csv(r'Zillow\properties_2016.csv')
    print(df1.shape)
    df2 = pd.read_csv(r'Zillow\train_2016_v2.csv')
    print(df2.shape)

    df_train = df2.merge(df1, how='left', on='parcelid')

    #revisit this fillna
    df_train = df_train.fillna(0)

    df_train['taxdelinquencyflag'] = df_train.taxdelinquencyflag.apply(lambda x: 1 if x == 'Y' else 0 )
    df_train['fireplaceflag'] = df_train.hashottuborspa.astype(int)
    df_train['hashottuborspa'] = df_train.hashottuborspa.astype(int)

    #use all old data for training
    #split the new data, 50% train, 50% dev
    df_train_1 = df_train[df_train.transactiondate < '2016-10-15']
    df_train_2 = df_train[df_train.transactiondate >= '2016-10-15']

    train_index, dev_index = train_test_split(df_train_2.index, test_size=0.5, random_state=0)
    df_train = pd.concat([df_train_1, df_train_2.loc[train_index]])
    df_dev = df_train_2.loc[dev_index]

    y_train = df_train.logerror.values
    y_dev = df_dev.logerror.values

    df_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
    df_dev = df_dev.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)

    X_train = df_train.values
    X_dev = df_dev.values

    print(X_train.shape)
    print(X_dev.shape)

    #normalization - mean, var from train set, apply to dev set and test set
    scaler = preprocessing.StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)   
    X_dev = scaler.transform(X_dev)

    X_train_k = X_train
    X_dev_k = X_dev
    Y_train_k = y_train.reshape(y_train.shape[0], 1)
    Y_dev_k = y_dev.reshape(y_dev.shape[0], 1)

    X_train_dnn = X_train.T
    X_dev_dnn = X_dev.T
    Y_train_dnn = y_train.reshape(1, y_train.shape[0])
    Y_dev_dnn = y_dev.reshape(1, y_dev.shape[0])

    if api == 'dnn':
        return X_train_dnn, X_dev_dnn, Y_train_dnn, Y_dev_dnn
    elif api == 'keras':
        return X_train_k, X_dev_k, Y_train_k, Y_dev_k
    else:
        raise ValueError('api')

def zillow_dnn(parameters, X_train, X_dev, Y_train, Y_dev, random_seed=None):
    #random seed affects parameters init, batch shuffle
    if random_seed != None:
        np.random.seed(random_seed)

    hidden_1, hidden_2, activation, epochs, learning_rate, batch_size, method, momentum = parameters
    if activation == 'tanh':
        initializer = 'Lecun'
        activation = np.tanh
    elif activation == 'relu':
        initializer = 'He'
        activation = DNN.relu
    else:
        raise ValueError('activation')

    regularizer = None
    #regularizer = regularizers.l2(0.01)

    dnn = DNN.DNN()
    dnn.add_input_layer(X_train.shape[0])
    dnn.add_layer(hidden_1, activation)
    dnn.add_layer(hidden_2, activation)
    dnn.add_layer(1, DNN.identity)
    dnn.costfunction = DNN.mean_squared_error
    dnn.init = initializer
    if method == 'GD':
        optimizer = ('GD', momentum)
    else:
        optimizer = method
    dnn.optimizer = optimizer
    dnn.compile()

    gradient_check=False
    verbose = None

    results = dnn.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_dev, Y_dev)], eval_metric='mae', learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, gradient_check=gradient_check, verbose=verbose)

    Y_predict = dnn.predict(X_train)
    accuracy_train = mean_absolute_error(np.reshape(Y_train, Y_train.shape[1]), np.reshape(Y_predict, Y_predict.shape[1]))
    Y_predict = dnn.predict(X_dev)
    accuracy_dev = mean_absolute_error(np.reshape(Y_dev, Y_dev.shape[1]), np.reshape(Y_predict, Y_predict.shape[1]))

    print(accuracy_train, accuracy_dev, hidden_1, hidden_2, activation, epochs, learning_rate, batch_size, method, momentum)

def zillow_keras(parameters, X_train, X_dev, Y_train, Y_dev, random_seed=None):
    #random seed
    if random_seed != None:
        np.random.seed(random_seed)
    
    hidden_1, hidden_2, activation, epochs, learning_rate, batch_size, method, momentum = parameters

    if activation == 'tanh':
        initializer = keras.initializers.lecun_normal()
    elif activation == 'relu':
        initializer = keras.initializers.he_normal()
    else:
        raise ValueError('activation')

    regularizer = None
    #regularizer = regularizers.l2(0.01)

    model = Sequential()
    model.add(Dense(hidden_1, input_dim=X_train.shape[1], kernel_initializer=initializer, activation=activation, kernel_regularizer=regularizer))
    model.add(Dense(hidden_2, kernel_initializer=initializer, activation=activation, kernel_regularizer=regularizer))
    model.add(Dense(1, kernel_initializer=initializer, kernel_regularizer=regularizer))

    if method == 'GD':
        optimizer = optimizers.SGD(lr=learning_rate, momentum=momentum)
    elif method == 'RMSProp':
        optimizer = optimizers.RMSprop(lr=learning_rate)
    elif method == 'Adam':
        optimizer = optimizers.Adam(lr=learning_rate)
    else:
        raise ValueError('method')

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    initial_epoch = 0
    verbose = 0
    #verbose2 = 10
    verbose2 = None
    results = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=[KerasCallback(verbose2)], validation_data = (X_dev, Y_dev), verbose=verbose, initial_epoch=initial_epoch)

    Y_predict = model.predict(X_train)
    try:
        accuracy_train = mean_absolute_error(Y_train, Y_predict)
    except ValueError:
        accuracy_train = 0
    Y_predict = model.predict(X_dev)
    try:
        accuracy_dev = mean_absolute_error(Y_dev, Y_predict)
    except ValueError:
        accuracy_dev = 0

    print(accuracy_train, accuracy_dev, hidden_1, hidden_2, activation, epochs, learning_rate, batch_size, method, momentum)


np.random.seed(0) 
#titanic
X_train, X_dev, Y_train, Y_dev = titanic_prep('dnn')
titanic_dnn(X_train, X_dev, Y_train, Y_dev)
parameters = (20, 10, 'tanh', 10000, 0.001, 'Adam', 0)
X_train, X_dev, Y_train, Y_dev = titanic_prep('keras')
titanic_keras(parameters, X_train, X_dev, Y_train, Y_dev)

#MNIST
parameters = (200, 50, 'tanh', 100, 0.1, 256, 'GD', 0.9)
X_train, X_dev, Y_train, Y_dev = MNIST_prep('dnn')
MNIST_dnn(parameters, X_train, X_dev, Y_train, Y_dev)
X_train, X_dev, Y_train, Y_dev = MNIST_prep('keras')
MNIST_keras(parameters, X_train, X_dev, Y_train, Y_dev)

#Zillow
parameters = (50, 50, 'relu', 500, 0.01, 512, 'RMSProp', 0)
X_train, X_dev, Y_train, Y_dev = zillow_prep('dnn')
zillow_dnn(parameters, X_train, X_dev, Y_train, Y_dev)
#zillow_tune(X_train, X_dev, Y_train, Y_dev, 'dnn')
X_train, X_dev, Y_train, Y_dev = zillow_prep('keras')
zillow_keras(parameters, X_train, X_dev, Y_train, Y_dev)
#zillow_tune(X_train, X_dev, Y_train, Y_dev, 'keras')

