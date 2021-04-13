import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid(5))
print(sigmoid(0.8))
a = np.full(5, 0.5)
print(a)
print(round(0.99999999, 3))

# b = [1, 1, 1] - 3
# print(b)
print(sigmoid(5.008))
print(sigmoid(0.843))

x1 = np.array([[1, 1, 1],
               [0, 0, 1],
               [0, 0, 1]])

x2 = np.array([[1, 0, 0],
               [1, 0, 0],
               [1, 1, 1]])

x3 = np.array([[1, 1, 1],
               [1, 0, 0],
               [1, 1, 1]])

d1 = np.array([1, 0, 0])
d2 = np.array([0, 1, 0])
d3 = np.array([0, 0, 1])

X = [x1, x2, x3]
D = [d1, d2, d3]
A = [0.0 for i in range(2)]
Z = [0.0 for i in range(2)]
O = np.full((3, 3), 0.0)
W1 = np.full((9, 2), 1.0)
W2 = np.full((2, 3), 1.0)

print(W1)
print(W1[:, 0])

for i in range(3):
    for k in range(3):
        A[k] = np.dot(X[i], W1[:, k])
        Z[k] = sigmoid(A[k])