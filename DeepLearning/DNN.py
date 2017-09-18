import numpy as np
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

'''

binary classification
    [linear->activation]->[linear->activation]->...->[linear->sigmoid]
    activation: tanh, relu
    Loss: cross entropy

classification
    [linear->activation]->[linear->activation]->...->[linear->softmax]
    activation: tanh, relu
    Loss: categorical cross entropy

regression
    [linear->activation]->[linear->activation]->...->[linear->identity]
    activation: tanh, relu
    Loss: MSRE

Hyper parameters
* number of hidden layers, number of nodes
* parameters initialization (uniform random, normal random, etc.)
    big difference in Titanic/Keras
* activation functions
* learning rate

Fixed parameters
* loss function

'''

###
### TODO
### dropout
###

#
#activation functions
#
def identity(Z):
    return Z

def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    return A

def softmax(Z):
    A = np.exp(Z)/np.sum(np.exp(Z), axis=0)
    return A
#
#cost functions
#

def binary_crossentropy(AL, Y):
    m = Y.shape[1] # number of examples
    logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y))
    cost = -np.sum(logprobs)/m
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect, e.g. turns [[17]] into 17 
    assert(isinstance(cost, float))
    return cost

def categorical_crossentropy(AL, Y):
    m = Y.shape[1] # number of examples
    logprobs = np.multiply(np.log(AL), Y)
    cost = -np.sum(np.sum(logprobs, axis=0))/m
    assert(isinstance(cost, float))
    return cost

def mean_squared_error(AL, Y):
    m = Y.shape[1] # number of examples
    cost = np.sum((AL - Y)*(AL - Y))/m
    assert(isinstance(cost, float))
    return cost

class DNN:
    '''
    L = number of layers
    n = number of nodes in each layer. array[L + 1]
    W, b = parameters. array[L + 1], 0 not used
    activations = activation functions. array[L + 1], 0 not used
    costfunction = the cost(loss) function
    init = parameters initalization method

    lambd = regularization factor. global for now, can be defined per layer
    '''
    def __init__(self):
        self.L = -1
        self.n = None
        self.W = None
        self.b = None
        self.activations = None
        self.costfunction = None
        self.init = 'RandomNormal'
        self.lambd = 0
        
    def add_input_layer(self, node_count):
        assert(self.L == -1)
        self.L = 0
        self.n = [node_count]
        self.activations = [None]
    
    def add_layer(self, node_count, activation):
        assert(activation.__name__ in ['sigmoid', 'identity', 'tanh', 'relu', 'softmax'])
        self.L += 1
        self.n.append(node_count)
        self.activations.append(activation)

    def compile(self):
        '''
        Checks the member variables
        '''
        if self.L < 2:
            raise ValueError('L')
        if self.n == None or len(self.n) != self.L + 1:
            raise ValueError('n')
        if self.activations == None or len(self.activations) != self.L + 1:
            raise ValueError('activations')
        if self.costfunction == None:
            raise ValueError('costfunction')
        assert(self.costfunction.__name__ in ['binary_crossentropy', 'categorical_crossentropy', 'mean_squared_error'])
        if self.lambd < 0 or self.lambd >= 1:
            raise ValueError('costfunction')

        self.__initialize_parameters()

    def __initialize_parameters(self):
        '''
        Initializes parameters W and b
        '''
        self.W = []
        self.b = []
        
        self.W.append(None)
        self.b.append(None)

        for i in range(1, self.L + 1):
            if self.init == 'RandomNormal':
                Wi = np.random.randn(self.n[i], self.n[i - 1]) * 0.01
            elif self.init == 'He':
                if i < self.L and self.activations[i].__name__ != 'relu':
                    print('Warning: He works well with relu')
                Wi = np.random.randn(self.n[i], self.n[i - 1]) *np.sqrt(2./self.n[i - 1])
            elif self.init == 'Lecun':
                if i < self.L and self.activations[i].__name__ != 'tanh':
                    print('Warning: Lecun works well with tanh')
                Wi = np.random.randn(self.n[i], self.n[i - 1]) *np.sqrt(1./self.n[i - 1])
            else:
                raise ValueError('init')

            bi = np.zeros((self.n[i], 1))
            self.W.append(Wi)
            self.b.append(bi)

    def __forward_propagation(self, X):
        """
        Forward propagation
        
        Argument:
        X -- input data of size (n[0], m)

        Returns:
        A -- The output of the last activation
        cache -- a list that contains (Z[i], A[i]) for i in 0..L
        """

        cache = [(None, X)]
        A = X
        for i in range(1, self.L + 1):
            Z = np.dot(self.W[i], A) + self.b[i]
            A = self.activations[i](Z)
            cache.append((Z, A))
            
        assert(A.shape == (self.n[self.L], X.shape[1]))
        return A, cache

    def __calculate_cost(self, A, Y):
        m = Y.shape[1]

        cost = self.costfunction(A, Y)
        #for i in range(1, self.L + 1):
        #    cost += self.lambd/m/2*np.sum(np.square(self.W[i]))
        cost += self.lambd/m/2*np.sum([np.sum(np.square(x)) for x in self.W[1:]])

        return cost

    def __backward_propagation(self, cache, X, Y):
        """
        Implement the backward propagation.

        Arguments:
        cache -- a list that contains (Z[i], A[i]) for i in 0..L
        X -- input data of shape (n[0], m)
        Y -- true output of shape (n[L], m)

        Returns:
        grads -- list of gradients(dW[i], db[i]) for i in 0..L
        """
        m = X.shape[1]

        grads = []
        # Backward propagation: calculate dW[i], db[i]
        for i in reversed(range(1, self.L + 1)):
            Z, A = cache[i]
            Z_prev, A_prev = cache[i - 1]
            
            if i == self.L:
                if self.activations[i].__name__ == 'sigmoid':
                    assert(self.costfunction.__name__ == 'binary_crossentropy')
                    dZ = A - Y
                elif self.activations[i].__name__ == 'softmax':
                    assert(self.costfunction.__name__ == 'categorical_crossentropy')
                    dZ = A - Y 
                elif self.activations[i].__name__ == 'identity':
                    assert(self.costfunction.__name__ == 'mean_squared_error')
                    dZ = 2*(A - Y)
                else:
                    raise Exception('unknown activation')
            else:
                if self.activations[i].__name__ == 'tanh':
                    dZ = np.dot(self.W[i + 1].T, dZ)*(1 - np.power(A, 2))
                elif self.activations[i].__name__ == 'relu':
                    dZ = np.dot(self.W[i + 1].T, dZ)
                    dZ[Z < 0] = 0
                else:
                    raise Exception('unknown activation')        
                
            if dZ.shape != (self.n[i], m):
                raise Exception('dZ')

            dW = np.dot(dZ, A_prev.T)/m
            dW += self.lambd/m*self.W[i]
            db = np.sum(dZ, axis=1, keepdims=True)/m
            grads.append((dW, db))
            
        #index 0 is not used
        grads.append((None, None))
        return list(reversed(grads))

    def __update_parameters(self, grads, learning_rate):
        '''
        Update parameters
        '''
        for i in range(1, self.L + 1):
            dW, db = grads[i]
            self.W[i] -= learning_rate*dW
            self.b[i] -= learning_rate*db
            
    def __weights_to_array(self, grads):
        num_parameters = 0
        for i in range(1, self.L + 1):
            num_parameters += self.n[i] * self.n[i - 1] + self.n[i]
        theta = np.zeros(num_parameters)
        dtheta = np.zeros(num_parameters)
        start = 0
        for i in range(1, self.L + 1):
            np.put(theta, start + np.arange(self.n[i] * self.n[i - 1]), self.W[i].reshape(-1))
            np.put(dtheta, start + np.arange(self.n[i] * self.n[i - 1]), grads[i][0].reshape(-1))
            start += self.n[i] * self.n[i - 1]
            np.put(theta, start + np.arange(self.n[i]), self.b[i].reshape(-1))
            np.put(dtheta, start + np.arange(self.n[i]), grads[i][1].reshape(-1))
            start += self.n[i]
        assert(len(theta) == len(dtheta))
        return theta, dtheta

    def __weights_from_array(self, theta):
        start = 0
        for i in range(1, self.L + 1):
            self.W[i] = theta[start : start + self.n[i] * self.n[i - 1]].reshape(self.W[i].shape)
            start += self.n[i] * self.n[i - 1]
            self.b[i] = theta[start : start + self.n[i]].reshape(self.b[i].shape)
            start += self.n[i]

        assert(start == len(theta))

    def __gradient_check(self, X, Y, grads, num_parameters=0, epsilon = 1e-7):
        theta, dtheta = self.__weights_to_array(grads)

        if num_parameters == 0:
            num_parameters = len(theta)
        gradapprox = np.zeros(num_parameters)
    
        print('gradient check parameters', num_parameters, '/', len(theta))
        for i in range(num_parameters):
            theta[i] += epsilon
            self.__weights_from_array(theta)
            A, _ = self.__forward_propagation(X)
            J_plus = self.__calculate_cost(A, Y)

            theta[i] -= 2*epsilon
            self.__weights_from_array(theta)
            A, _ = self.__forward_propagation(X)
            J_minus = self.__calculate_cost(A, Y)

            theta[i] += epsilon

            gradapprox[i] = (J_plus - J_minus)/2/epsilon

            if i % 10 == 0:
                print('.', end='', flush=True)

        print('')

        self.__weights_from_array(theta)

        numerator = np.linalg.norm(dtheta[:num_parameters] - gradapprox)                     
        denominator = np.linalg.norm(dtheta[:num_parameters]) + np.linalg.norm(gradapprox)   
        difference = numerator/denominator                                

        if difference > 1e-7:
            print('gradient check error')

        print('gradient check done', difference)

    @staticmethod
    def __create_mini_batches(X, Y, batch_size = 64):
        """
        Creates a list of random minibatches from (X, Y)
    
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        batch_size -- size of the mini-batches, integer
    
        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
    
        # number of training examples
        m = X.shape[1]                  
        mini_batches = []
        
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]#.reshape((1,m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/batch_size) # number of mini batches of size batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k*batch_size : (k+1)*batch_size]
            mini_batch_Y = shuffled_Y[:, k*batch_size : (k+1)*batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    
        # Handling the end case (last mini-batch < batch_size)
        if m % batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches*batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches*batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    
        return mini_batches

    def fit(self, X, Y, eval_set=None, eval_metric='error', learning_rate=0.1, num_iterations=10000, batch_size=256, verbose=None, gradient_check=False, num_parameters=0):
        '''
        Fit
        '''

        results = {'loss': []}
        if eval_set == None:
            eval_set = []
        for i in range(len(eval_set)):
            results['eval' + str(i)] = []

        # number of training examples
        m = X.shape[1]

        # Loop (epochs)
        for i in range(0, num_iterations):
            epoch_cost = 0

            minibatches = DNN.__create_mini_batches(X, Y, batch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation.
                A, cache = self.__forward_propagation(minibatch_X)

                # Cost function. 
                cost = self.__calculate_cost(A, minibatch_Y)
                results['loss'].append(cost)
                epoch_cost += cost*minibatch_X.shape[1]

                # Backpropagation
                grads = self.__backward_propagation(cache, minibatch_X, minibatch_Y)

                if gradient_check == True:
                    self.__gradient_check(minibatch_X, minibatch_Y, grads, num_parameters=num_parameters)

                # Gradient descent parameter update. 
                self.__update_parameters(grads, learning_rate)

                if verbose != None and i % verbose == 0:
                    print('.', end='', flush=True)

            epoch_cost /= m
            epoch_costs = [epoch_cost]

            i_eval = 0
            for X_eval, Y_eval in eval_set:
                predictions = self.predict(X_eval)
                if eval_metric == 'error':
                    if predictions.shape[0] == 1:
                        #classifies to 0/1 using 0.5 as the threshold.
                        predictions = predictions > 0.5
                        cost = accuracy_score(np.reshape(Y_eval, Y_eval.shape[1]), np.reshape(predictions, predictions.shape[1]))
                    else:
                        cost = accuracy_score(predictions.argmax(axis=0), Y_eval.argmax(axis=0))
                    results['eval' + str(i_eval)].append(cost)
                    epoch_costs.append(cost)
                elif eval_metric == 'mae':
                    cost = mean_absolute_error(np.reshape(Y_eval, Y_eval.shape[1]), np.reshape(predictions, predictions.shape[1]))
                    results['eval' + str(i_eval)].append(cost)
                    epoch_costs.append(cost)
                i_eval += 1

            # Print the cost 
            if verbose != None and i % verbose == 0:
                print('')
                print ("Cost after iteration", i, ["{0:0.6f}".format(i) for i in epoch_costs])
                #print ("Cost after iteration", i, epoch_costs)
                
        return results            
    
    def predict(self, X):
        """
        Using the learned parameters, predicts the output for each example in X

        Arguments:
        X -- input data of size (n[0], m)

        Returns
        predictions -- vector of predicted outputs
        """

        A, _ = self.__forward_propagation(X)
        return A

