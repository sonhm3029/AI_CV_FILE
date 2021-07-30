import numpy as np
from Layer import Layer
"""
    NN: is a simple neural network model for classification & regression problems
    ....

    Attributes
    ----------
        X:  type -> np.ndarray
            the input data
        Y: type -> np.ndarray
            the target data
        output_activation: type-> string
            the activation function of the last layer,
            the output layer
            default -> 'sigmoid'
    
    Example
    -------
    > from NN import nn
    > from Layer import Layer
    > # import some data from sklearn library
    > from sklearn.datasets import load_breast_cancer
    > inputs = data.data
    > targets = data.target.reshape(-1,1)
    > neural_network_model = nn(inputs, targets)
    > # add hidden layers
    > neural_network_model.add_layer( Layer(32, activation='relu') )
    > neural_network_model.fit()
    > # predict data
    > Y_pred = neural_network_model.predict(INPUTS)
    > # plot cost function
    > import matplotlib.pyplot as plt
    > plt.plot(neural_network_model._costs)
    > plt.show()

"""
class NN:

    def __init__(self, X, Y, output_activation='sigmoid'):
        self._X = X 
        self._Y = Y
        self._layers = []
        self._output_activation = output_activation

    
    
    def add_layer(self, layer):
        if not isinstance(layer, Layer):
            raise Exception("Invalid Type", type(layer), " != <class 'Layer'>")
        self._layers.append(layer)

    

    # Train data
    def fit(self, learning_rate=0.01, iteration=1000):
        self._setup()
        self._costs = []
        self._learning_rate = learning_rate
        self._iteration = iteration
        for i in range(iteration):
            self._fowardPropagation()
            self._backPropagation()
            print(self._calc_cost(self._layers[len(self._layers)-1].values))
            if(i%100 == 0):
                self._costs.append( self._calc_cost(self._layers[len(self._layers)-1].values) )
        

    # return the cost function
    def _calc_cost(self, Y_pred):
        return np.sum( np.square(self._Y - Y_pred) / 2 )
    
    # configuration the shape,
    # weight and bias of each layer
    # add output layer
    def _setup(self):
        for index, layer in enumerate(self._layers):
            if(index == 0): # first hidden layer
                layer._setup(self._X)
            else:
                layer._setup(self._layers[index-1])
        ### setup and add output layer
        output_layer = Layer( self._Y.shape[1], activation=self._output_activation)
        output_layer._setup( self._layers[len(self._layers)-1] )
        self.add_layer(output_layer)


    def _fowardPropagation(self):
        for index, layer in enumerate(self._layers):
            if(index == 0): # first hidden layer
                layer._foward(self._X)
            else:
                layer._foward(self._layers[index-1])


    def _backPropagation(self):
        delta = self._Y - self._layers[len(self._layers)-1].values
        for i in range(len(self._layers)-1, -1, -1):
            if (i == 0): # first hidden layer
                delta = self._layers[i]._backward(delta, self._X, self._learning_rate)
            else:
                delta = self._layers[i]._backward(delta, self._layers[i-1], self._learning_rate)


    def predict(self, X_test):
        for index, layer in enumerate(self._layers):
            if(index == 0):
                layer._foward(X_test)
            else:
                layer._foward(self._layers[index-1])
        if self._is_continues(): # if target labels is continues
            return self._layers[ len(self._layers)-1 ].values
        if self._is_multiclass(): # if target labels is multiclass
            return self._threshold_multiclass( self._layers[ len(self._layers)-1 ] )
        return self._threshold( self._layers[ len(self._layers)-1 ], 0.5 ) # binary classification


    # set the 'predict.value' > 'value' [treshhold] to '1' others to '0'
    def _threshold(self, target, value):
        predict = target.values
        predict[predict<value] = 0
        predict[predict>=value] = 1
        return predict

    # set the max 'predict.value' to '1' others to '0'
    def _threshold_multiclass(self, target):
        predict = target.values
        predict = np.where(predict==np.max(predict, keepdims=True, axis=1), 1, 0  )
        # predict[] = 1 | 0
        return predict

    # check if it's a multiclassfication problem
    def _is_multiclass(self):
        return len(np.unique(self._Y)) > 2

    # check if it's a regression problem
    def _is_continues(self):
        return len(np.unique(self._Y)) >  (self._Y.shape[0] / 3 )
        