import numpy as np

# n = 10

# array = np.zeros(n)

# size_even = array[::2].shape[0]
# array[::2] = np.arange(2,2 - 0.5 * size_even,-0.5)
# odd_id = np.arange(1,array.shape[0],2)
# array[odd_id] = -1
# print(array)

# x = np.arange(0,10,1)
# y = np.zeros(10)
# z = np.zeros(10)
# for i in range(0,10):
#     y[i] = np.pi/2 - x[i]
#     z[i] = np.cos(x[i]) - np.sin(x[i])

# print("y = ", y)
# print("z = ", z)
# print(np.sum(z))

x = np.arange(10)
print(x)
print(x.max())
print(x.argmax())