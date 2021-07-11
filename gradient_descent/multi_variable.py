# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(2)

#create 1000 data point near y = 4 + 3x, display it.
X = np.random.rand(1000, 1)
print(X.shape)
y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added


# Building Xbar 
one = np.ones((X.shape[0],1))
print(X)
Xbar = np.concatenate((one, X), axis = 1)
print(Xbar)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula: w = ',w_lr.T)


# Display result
w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
print(x0)
y0 = w_0 + w_1*x0
print(y0)

# Draw the fitting line 
plt.plot(X.T, y.T, 'b.')     # data 
plt.plot(x0, y0, 'y', linewidth = 2)   # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()

def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2;

def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break 
        w.append(w_new)
    return (w, it) 

w_init = np.array([[2], [1]])
print("w_init shape ", w_init.shape)
(w1, it1) = myGD(w_init, grad, 1)
print(w[-1].shape)
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1))