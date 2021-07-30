from sys import path
path.append('..')
import numpy as np
from NN import NN
from Layer import Layer
import os
import cv2
from sklearn.datasets import load_breast_cancer # binaryclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


path_human = 'C:\\Users\pC\OneDrive\Desktop\desktop\AI_CV\\task_7_demo_NN\\task_7_demo_NN\\Human'
part_non_human = 'C:\\Users\pC\OneDrive\Desktop\desktop\AI_CV\\task_7_demo_NN\\task_7_demo_NN\\Non-Human'
ones_human = np.ones((64*64, 1)).reshape(1, -1)
print(ones_human)
for file in os.listdir(path_human):
  path_img_human = path_human+'/'+file
  image_human = cv2.imread(path_img_human, 0).reshape(1, -1)
  ones_human = np.vstack((ones_human, image_human))

data_human = np.delete(ones_human, 0, 0)
N, d = data_human.shape
print(N,d)
ones_non_human = np.ones((64*64, 1)).reshape(1, -1)
for file in os.listdir(part_non_human):
  path_img_non_human = part_non_human+'/'+file
  image_non_human = cv2.imread(path_img_non_human, 0).reshape(1, -1)
  ones_non_human = np.vstack((ones_non_human, image_non_human))

data_non_human = np.delete(ones_non_human, 0, 0)

feature_set = np.vstack((data_human,data_non_human))

label_ones = np.ones((120,1))
label_zeros = np.zeros((120,1))
targets = np.vstack((label_ones,label_zeros))

X_train, X_test, Y_train, Y_test = train_test_split(feature_set, targets, test_size=0.2)

print(X_train.shape)
# create the network
nn_model = NN(X_train, Y_train)
nn_model.add_layer( Layer(24, activation='relu' ) )
nn_model.add_layer( Layer(12, activation='sigmoid') )


#fit the network
nn_model.fit(iteration=1000, learning_rate=0.001)

# plot cost function

Y_train_pred = nn_model.predict(X_train)
Y_test_pred = nn_model.predict(X_test)


# print(Y_test_pred )
