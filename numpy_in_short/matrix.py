import numpy as np

# A = np.arange(1,10,1)
# A = A.reshape(3,3)
# x = np.eye(3)
# print(A)
# print(np.diag(A))

# n = np.arange(1,10,1)

# A = np.diag(n, k = -1)
# print(A)

# A = np.array([[1,2],[3,4]])
# def myfunc(A):
#     sum = 0
#     num_row = A.shape[0]
#     for i in range(0, num_row):
#         sum += np.sum(A[i,::2])
#     return sum

x = np.arange(0,8,1)
y = np.arange(9,17,1)
z = np.arange(4,12,1)
B = np.array([x,y,z])

# def myfunc(A):
#     return np.sum(A[::2][:,::2])
A = np.array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]], dtype= float)
# print(A[::2][::2])
# print(myfunc(A))
# def myfunc(A):
#     return np.sum(np.max(A, axis = 1) - np.min(A, axis = 1))

# A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# C = A*A
# def norm_fro(A):
#     return np.sqrt(np.sum(A*A))
# print(norm_fro(A))
# print(np.linalg.norm(A))
# A = np.arange(1,13)
# A = np.reshape(A.reshape(4,3, order = 'F'),(3,4))
# print(A)

# def zero_mean(A):
#     N = A.shape[0]
#     d = A.shape[1]
#     B = np.sum(A, axis = 1)/d
#     print(B)
#     for i in range(0,N):
#         A[i,:] = A[i,:] - B[i]  
#         #print(B[i])
#     return A
# print(A)
# C = zero_mean(A)
# print(C)

b = np.array([1,2,3])
print(b.shape)



