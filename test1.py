import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid(5))
print(sigmoid(0.8))
a = np.full(5, 0.5)
print(a)
print(round(0.99999999, 3))