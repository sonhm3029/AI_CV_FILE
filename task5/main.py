from __future__ import unicode_literals, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    return 1/(1+ np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)* ( 1- sigmoid(x))

class Neural_NetWork:

    #class function
    def __init__(self):
      #input layer:
      input_data = pd.read_csv("C:\\Users\pC\OneDrive\Desktop\desktop\AI_CV\\task5\\training.csv").values
      N, d = input_data.shape
      self.input_layer = input_data[:,0:8].reshape(-1,8)
      self.output_layer = input_data[:,8].reshape(-1,1)

      #init weight and bias
      # 5 W weights each have 8 w for the hidden layer
      # 1 W have 5 w for the output layer
      self.hidden_weight = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ,0.1, 0.1],
                                    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3 ,0.3, 0.3],
                                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ,0.5, 0.5],
                                    [0.7, 0.7, 0.7, 0.7, 0.7, 0.7 ,0.7, 0.7],
                                    [0.9, 0.9, 0.9, 0.9, 0.9, 0.9 ,0.9, 0.9]]).reshape(8,-1)
      self.output_weight = np.array([0.2, 0.4, 0.6, 0.8, 0.9]).reshape(5,-1)

      #learning rate
      self.eta = 0.01

    def _training_(self):
        
        for epoch in range(500):
            #Input for hidden layer
            input_hidden = np.dot(self.input_layer, self.hidden_weight) 

            #Ouput from hidden layer
            output_hidden = sigmoid(input_hidden)
        
            #input for ouput layer
            input_op = np.dot(output_hidden, self.output_weight)

            #Ouput from output layer
            output_predict = sigmoid(input_op)



            self.output_weight -= self.eta * np.dot(output_hidden.T, output_predict - self.output_layer)
            self.hidden_weight -= self.eta * np.dot(self.input_layer.T, output_hidden - self.output_layer)

    def training(self):
        bias = 0
        for epoch in range(100000):
            #Input for hidden layer
            input_hidden = np.dot(self.input_layer, self.hidden_weight)

            #Ouput from hidden layer
            output_hidden = sigmoid(input_hidden)
        
            #input for ouput layer
            input_op = np.dot(output_hidden, self.output_weight)

            #Ouput from output layer
            output_op = sigmoid(input_op)
            #============================================================
            # Phase 1
            # Caculating Mean Squared Error
            error_out= ((1/2) * (np.power((output_op - self.output_layer), 2)))


            #Derivatives for phase 1
            derror_douto = output_op - self.output_layer
            douto_dino = sigmoid_derivative(input_op)
            dino_dwo = output_hidden

            derror_dwo = np.dot(dino_dwo.T, derror_douto * douto_dino)

            #==============================================================
            # Phase 2
            #
            #derror_w1 = derror_douth * douth_dinh * dinh_dw1
            #derror_douth = derror_dino * dino_outh

            #Derivatives for phase 2
            derror_dino = derror_douto * douto_dino
            dino_douth = self.output_weight
            derror_douth = np.dot(derror_dino, dino_douth.T)
            douth_dinh = sigmoid_derivative(input_hidden)
            dinh_dwh = self.input_layer
            derror_wh = np.dot(dinh_dwh.T, douth_dinh * derror_douth)

            #Update weight
            self.hidden_weight -= self.eta * derror_wh
            self.output_weight -= self.eta * derror_dwo
            #bias -= self.eta*epoch













neu1 = Neural_NetWork()
neu1.__init__()
print(neu1.input_layer.shape)
print(neu1.hidden_weight.shape)
neu1.training()
print(neu1.hidden_weight)


test = pd.read_csv("C:\\Users\pC\OneDrive\Desktop\desktop\AI_CV\\task5\\testing.csv").values
in_key = test[:,0:8].reshape(-1,8)
test_key = test[:,8].reshape(-1,1)
o_in = sigmoid(np.dot(in_key, neu1.hidden_weight))
out = sigmoid(np.dot(o_in, neu1.output_weight))

for i in range(len(out)):
    if( out[i] < 0.5):
        out[i] = 0
    else:
        out[i] = 1

count = 0
for i in range(len(out)):
    if(out[i] == test_key[i]):
        count +=1


print(count)
print(out.shape)







# Create data from dataset
# input_data = pd.read_csv("C:\\Users\pC\OneDrive\Desktop\desktop\AI_CV\\task5\\training.csv").values
# N, d = input_data.shape
# input_layer = input_data[:,0:8].reshape(-1, 8)
# output_layer = input_data[:,8].reshape(-1,1)
# print(output_layer.shape)
# #add bias
# x = np.hstack((np.ones((N,1)), input_layer))

# w = np.array([0,0.1,0.1,0.1,0.1, 0.1, 0.1, 0.1, 0.1]).reshape(-1,1)



