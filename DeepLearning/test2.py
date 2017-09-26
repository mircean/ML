#
#TODO
#keras consistent results
#variance caused by random seed 
#variance caused by train/test split
#

#works only with keras for now
def titanic_tune(X_train, X_dev, Y_train, Y_dev):
    hidden_1_values = [10, 20]
    hidden_2_values = [10, 20]
    activations = ['tanh', 'relu']
    epochs_values = [10000, 100000]
    learning_rates = [0.1, 0.01, 0.001]
    methods = ['GD', 'RMSProp', 'Adam']
    momentum_values = [0, 0.9, 0.98]

    configs = []
    for hidden_1 in hidden_1_values:
        for hidden_2 in hidden_2_values:
            for activation in activations:
                for epochs in epochs_values:
                    for learning_rate in learning_rates:
                        for method in methods:
                            if method == 'GD':
                                for momentum in momentum_values:
                                    configs.append([hidden_1, hidden_2, activation, epochs, learning_rate, method, momentum])
                            else:
                                configs.append((hidden_1, hidden_2, activation, epochs, learning_rate, method, 0))


    print(len(configs))
    for parameters in configs:
        np.random.seed(0) # set a seed so that the results are consistent
        titanic_keras(parameters, X_train, X_dev, Y_train, Y_dev)

#works only with keras for now
def MNIST_tune(X_train, X_dev, Y_train, Y_dev):
    hidden_1_values = [100, 200]
    hidden_2_values = [20, 50]
    activations = ['tanh', 'relu']
    #epochs_values = [100, 333, 1000]
    epochs_values = [100]
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [256, 512, 1024]
    methods = ['GD', 'RMSProp', 'Adam']
    momentum_values = [0, 0.9, 0.98]

    configs = []
    for hidden_1 in hidden_1_values:
        for hidden_2 in hidden_2_values:
            for activation in activations:
                for epochs in epochs_values:
                    for learning_rate in learning_rates:
                        for batch_size in batch_sizes:
                            for method in methods:
                                if method == 'GD':
                                    for momentum in momentum_values:
                                        configs.append([hidden_1, hidden_2, activation, epochs, learning_rate, batch_size, method, momentum])
                                else:
                                    configs.append((hidden_1, hidden_2, activation, epochs, learning_rate, batch_size, method, 0))


    print(len(configs))
    for parameters in configs:
        np.random.seed(0) # set a seed so that the results are consistent
        MNIST_keras(parameters, X_train, X_dev, Y_train, Y_dev)
        
def zillow_tune(X_train, X_dev, Y_train, Y_dev, api='dnn'):
    hidden_1_values = [50, 100]
    hidden_2_values = [20, 50]
    activations = ['tanh', 'relu']
    #epochs_values = [100, 333, 1000]
    epochs_values = [500]
    learning_rates = [0.05, 0.01, 0.005, 0.001]
    #batch_sizes = [256, 512, 1024]
    batch_sizes = [256, 512]
    methods = ['GD', 'RMSProp', 'Adam']
    #momentum_values = [0, 0.9, 0.98]
    momentum_values = [0, 0.9]

    configs = []
    for hidden_1 in hidden_1_values:
        for hidden_2 in hidden_2_values:
            for activation in activations:
                for epochs in epochs_values:
                    for learning_rate in learning_rates:
                        for batch_size in batch_sizes:
                            for method in methods:
                                if method == 'GD':
                                    for momentum in momentum_values:
                                        configs.append([hidden_1, hidden_2, activation, epochs, learning_rate, batch_size, method, momentum])
                                else:
                                    configs.append((hidden_1, hidden_2, activation, epochs, learning_rate, batch_size, method, 0))


    print('Configs count', len(configs))
    for parameters in configs:
        np.random.seed(0) 
        if api == 'dnn':
            zillow_dnn(parameters, X_train, X_dev, Y_train, Y_dev)
        elif api == 'keras':
            zillow_keras(parameters, X_train, X_dev, Y_train, Y_dev)
        else:
            raise ValueError('api')

