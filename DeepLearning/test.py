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

#DNN
import DNN

#keras
'''
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
'''

def titanic_prep():
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

    #df_train
    #df_predict

    X = df_train.values
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=test_size, random_state=0)

    #scale: X = (X - X.mean(axis=0))/np.sqrt(X.var(axis=0))
    #X = preprocessing.scale(X)

    #normalization - mean, var from train set, apply to dev set and test set
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)   
    X_dev = scaler.transform(X_dev)

    X_train_k = X_train
    X_dev_k = X_dev
    Y_train_k = y_train.reshape(y_train.shape[0], 1)
    Y_dev_k = y_dev.reshape(y_dev.shape[0], 1)

    X_train_my = X_train.T
    X_dev_my = X_dev.T
    Y_train_my = y_train.reshape(1, y_train.shape[0])
    Y_dev_my = y_dev.reshape(1, y_dev.shape[0])

    print(X_train.shape)
    print(y_train.shape)
    print(X_dev.shape)
    print(y_dev.shape)

    return X_train_my, X_dev_my, Y_train_my, Y_dev_my

def titanic_dnn(X_train_my, X_dev_my, Y_train_my, Y_dev_my):
    dnn = DNN.DNN()
    dnn.add_input_layer(X_train_my.shape[0])
    dnn.add_layer(10, np.tanh)
    dnn.add_layer(10, np.tanh)
    dnn.add_layer(1, DNN.sigmoid)
    dnn.costfunction = DNN.binary_crossentropy
    dnn.init = 'Lecun'
    dnn.compile()    
    
    np.random.seed(0) # set a seed so that the results are consistent

    learning_rate = 0.01
    num_iterations = 100
    verbose = 1000
    gradient_check = True
    #eval_set = None
    eval_set = [(X_train_my, Y_train_my), (X_dev_my, Y_dev_my)]
    results = dnn.fit(X_train_my, Y_train_my, eval_set=eval_set, eval_metric='error', learning_rate=learning_rate, num_iterations=num_iterations, verbose=verbose, gradient_check=gradient_check)

    Y_predict_my = dnn.predict(X_train_my)
    Y_predict_my = Y_predict_my > 0.5
    accuracy = accuracy_score(np.reshape(Y_train_my, Y_train_my.shape[1]), np.reshape(Y_predict_my, Y_predict_my.shape[1]))
    print('Accuracy train', accuracy)

    Y_predict_my = dnn.predict(X_dev_my)
    Y_predict_my = Y_predict_my > 0.5
    accuracy = accuracy_score(np.reshape(Y_dev_my, Y_dev_my.shape[1]), np.reshape(Y_predict_my, Y_predict_my.shape[1]))
    print('Accuracy dev', accuracy)

def MNIST_prep():
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

    X_train_k = X_train
    X_dev_k = X_dev

    enc = OneHotEncoder()
    Y_train_k = enc.fit_transform(y_train.reshape(-1, 1)).todense().A
    Y_dev_k = enc.transform(y_dev.reshape(-1, 1)).todense().A

    X_train_my = X_train_k.T
    X_dev_my = X_dev_k.T
    Y_train_my = Y_train_k.T
    Y_dev_my = Y_dev_k.T

    print(X_train.shape)
    print(y_train.shape)
    print(X_dev.shape)
    print(y_dev.shape)

    return X_train_my, X_dev_my, Y_train_my, Y_dev_my

def MNIST_dnn(X_train_my, X_dev_my, Y_train_my, Y_dev_my):
    dnn = DNN.DNN()
    dnn.add_input_layer(X_train_my.shape[0])
    dnn.add_layer(200, np.tanh)
    dnn.add_layer(50, np.tanh)
    dnn.add_layer(10, DNN.softmax)
    dnn.costfunction = DNN.categorical_crossentropy
    dnn.init = 'Lecun'
    #dnn.lambd = 0.1
    dnn.compile()

    np.random.seed(0) # set a seed so that the results are consistent

    learning_rate = 0.01
    num_iterations = 100
    mini_batch_size = 256
    verbose = 10
    gradient_check=False
    num_parameters = 2500
    
    #eval_set = None
    eval_set = [(X_train_my, Y_train_my), (X_dev_my, Y_dev_my)]
    results = dnn.fit(X_train_my, Y_train_my, eval_set=eval_set, eval_metric='error', learning_rate=learning_rate, num_iterations=num_iterations, mini_batch_size=mini_batch_size, gradient_check=gradient_check, num_parameters=num_parameters, verbose=verbose)

    Y_predict_my = dnn.predict(X_train_my)
    accuracy = accuracy_score(Y_predict_my.argmax(axis=0), Y_train_my.argmax(axis=0))
    print('Accuracy train', accuracy)

    Y_predict_my = dnn.predict(X_dev_my)
    accuracy = accuracy_score(Y_predict_my.argmax(axis=0), Y_dev_my.argmax(axis=0))
    print('Accuracy dev', accuracy)

#X_train_my, X_dev_my, Y_train_my, Y_dev_my = titanic_prep()
#titanic_dnn(X_train_my, X_dev_my, Y_train_my, Y_dev_my)

X_train_my, X_dev_my, Y_train_my, Y_dev_my = MNIST_prep()
MNIST_dnn(X_train_my, X_dev_my, Y_train_my, Y_dev_my)

'''
class MyCallback(keras.callbacks.Callback):
    def __init__(self, verbose):
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose != None and epoch % self.verbose == 0:
            print(epoch, logs['loss'], logs['acc'], logs['val_acc'])   

initializer = keras.initializers.RandomUniform(minval=0, maxval=0.01, seed=None)
model = Sequential()
model.add(Dense(200, input_dim=X_train_k.shape[1], kernel_initializer=initializer, activation='tanh'))
model.add(Dense(50, kernel_initializer=initializer, activation='tanh'))
model.add(Dense(10, kernel_initializer=initializer, activation='softmax'))
optimizer = optimizers.SGD(lr=0.1)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

np.random.seed(0) # set a seed so that the results are consistent
initial_epoch = 0
epochs = 1000
verbose = 0
verbose2 = 100
batch_size = int(X_train_k.shape[0])
results = model.fit(X_train_k, Y_train_k, batch_size=batch_size, epochs=epochs, callbacks=[MyCallback(verbose2)], validation_data = (X_dev_k, Y_dev_k), verbose=verbose, initial_epoch=initial_epoch)
Y_predict_k = model.predict(X_train_k)
print('Accuracy train', accuracy_score(Y_train_k.argmax(axis=1), Y_predict_k.argmax(axis=1)))
Y_predict_k = model.predict(X_dev_k)
print('Accuracy dev', accuracy_score(Y_dev_k.argmax(axis=1), Y_predict_k.argmax(axis=1)))
'''