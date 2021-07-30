import numpy as np

# from scipy.special import expit

# sigmoid function
def sigmoid():
    return lambda X: 1/ (1+np.exp(-X))

# sigmoid function derivative
def sigmoid_der():
    return lambda X: sigmoid()(X) * ( 1 - sigmoid()(X) )


# relu function
def relu():
    return lambda X: np.where(X >= 0, X, 0)

# relu function derivative
def relu_der():
    def _(X):
        X[X<=0] = 0
        X[X>0] =1
        return X
    return _

# softmax function
def softmax():
    def _(X):
        exps = np.exp(X)
        summ = np.sum(X, axis=0)
        return np.divide(exps, summ)
    return _

# softmax derivative
# <CHECK-LATER>
# def softmax_der():
#     pass

def no_func():
    return lambda X: X

def no_func_der():
    return lambda X: 1


def get_activation(activation):
    activation = activation.lower()
    if activation == 'sigmoid':
        return sigmoid(), sigmoid_der()
    elif activation == 'relu':
        return  relu(), relu_der()
    elif activation == 'no_func':
        return no_func(), no_func_der()
    # default
    return no_func(), no_func_der()

