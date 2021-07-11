from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

# #create data
# X = np.array([(0,0),(0,1),(1,0), (1,1)])
# y = np.array([0,0,0,1]).reshape(-1,1)


# one = np.ones((X.shape[0],1))
# X = np.concatenate((one,X), axis = 1)

# w_init = np.array([0., 0.1, 0.1]).reshape(-1,1)

# def sigmoid(s):
#     return 1/( 1 + np.exp(-s))



# def logistic_regression(X, y, w_init, learning_rate):
#     w = [w_init]
#     N = X.shape[0]
#     d = X.shape[1]
#     count = 0
#     check_w_after = 10
#     while count < 1000:
#         #mixdata
#         mix_id = np.random.permutation(N)
#         for i in mix_id:
#             xi = X[i,:].reshape(1,3)
#             yi = y[i].reshape(-1,1)
#             zi = sigmoid(xi.dot(w[-1]))
#             w_new = w[-1] + learning_rate*(xi.reshape(3,1).dot(yi-zi))
#             w.append(w_new)
#             count += 1
#             # print(xi.shape)
#             # print(yi.shape)
#             # print(zi.shape)
#             # print(w[-1].shape)
#             if count%check_w_after == 0:                
#                 if np.linalg.norm(w_new - w[-check_w_after]) < 1e-4:
#                     return w[-1]
#             break
#     return w[-1]



# print(logistic_regression(X,y,w_init,0.01))


# Cach 2
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#create data
X = np.array([(0,0),(0,1),(1,0), (1,1)]).reshape(-1,2)
y = np.array([0,0,0,1]).reshape(-1,1)
N = X.shape[0]

#Ve bang Scatter


#Them 1 vao cot du lieu
X = np.hstack((np.ones((N,1)),X))
w = np.array([0.,0.1,0.1]).reshape(-1,1)
# So lan lap buoc 2:
numOfIteration = 300
cost = np.zeros((numOfIteration,1))
learning_rate  = 0.01


for i in range(1, numOfIteration):

    y_predict = sigmoid(np.dot(X,w))
    w = w - learning_rate*np.dot(X.T, y_predict - y)


print(w)



