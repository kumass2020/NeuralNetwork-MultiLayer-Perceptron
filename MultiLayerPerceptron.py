import numpy as np
from math import ceil, floor


def init_network():
    # network = {}
    # network['']

    x1 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1]])
    d1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    x2 = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1]])
    d2 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    x3 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1]])
    d3 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    x4 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1]])
    d4 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    x5 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1]])
    d5 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    x6 = np.array([[1, 1, 0, 0, 0, 0, 1, 1],
                  [1, 1, 0, 0, 0, 0, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 0, 0, 0, 0, 1, 1],
                  [1, 1, 0, 0, 0, 0, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1]])
    d6 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    x7 = np.array([[0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1]])
    d7 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

    x8 = np.array([[0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 0, 0, 1, 1, 0, 0, 0]])
    d8 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    x9 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [1, 1, 1, 0, 0, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1]])
    d9 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

    x10 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1]])
    d10 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    # 가중치
    W1 = np.full((64, 5), 1.0)
    W2 = np.full((5, 10), 1.0)

    # # 은닉층 활성화 함수(Sigmoid) 전, 후
    # A = np.array([0, 0, 0, 0, 0])
    # Z = np.array([0, 0, 0, 0, 0])

    # X: 입력패턴, D: 출력패턴
    X = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    D = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]

    return X, D, W1, W2


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1.0 / (1.0 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout + (1.0 - self.out) * self.out
        return dx


# 입력층 : 64개
# 은닉층 : 5개
# 출력층 : 10개
sigmoid1 = Sigmoid()
sigmoid2 = Sigmoid()
offset = 0
W = 1
eta = 0.1
bias = 0.3
X, D, W1, W2 = init_network()
epoch = 0

# 은닉층 오차
# delta1 = np.array([0, 0, 0, 0, 0])
delta1 = np.full((10, 5), 0.0)
# 출력층 오차
# delta2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
delta2 = np.full((10, 10), 0.0)
A = [0.0 for i in range(5)]
Z = [0.0 for i in range(5)]
O = np.full((10, 10), 0.0)

test = X[0]
test = test.flatten()

while epoch < 2000:
    print("epoch: " + str(epoch))
    for i in range(10):  # ㄱ, ㄴ, ㄷ, ...
        # print("i: " + str(i))
        for k in range(5):  # A[0], A[1], ...
            # print("k: " + str(k))
            test4 = X[i].flatten()
            test5 = W1[:, k]
            # A.insert(k, np.dot(X[i].flatten(), W1[:, k]))
            A[k] = np.dot(X[i].flatten(), W1[:, k])

            # # Test
            # if k == 0:
            #     A[k] -= 30

            # print(A[0].shape)
            # print(type(A[0]))
            # print(type(Z))
            # print("")

            # test1 = sigmoid.forward(A[k])

            # Z.insert(k, sigmoid1.forward(A[k]))
            Z[k] = sigmoid1.forward(A[k])

            # Z = np.array(Z).flatten()
        test3 = Z
        for j in range(10):
            # print("j: " + str(j))
            test1 = np.array(Z).flatten()
            test2 = W2[:, j]
            # O.insert(j, np.dot(np.array(Z).flatten(), W2[:, j]))
            O[i][j] = np.dot(np.array(Z).flatten(), W2[:, j])
            O[i][j] = sigmoid2.forward(O[i][j])
            test6 = O[i][j]
            test7 = 1 - O[i][j]
            test8 = (D[i])[j] - O[i][j]
            delta2[j] = O[i][j] * (1 - O[i][j]) * ((D[i])[j] - O[i][j])
        for m in range(5):
            # print("m: " + str(m))
            summ = 0
            for n in range(10):
                # print("n: " + str(n))
                summ += delta2[n] * W2[m][n]
            delta1[m] = Z[m] * (1 - Z[m]) * summ
        # 역전파 1
        for n in range(5):
            for o in range(10):
                W2[n][o] = W2[n][o] + eta * delta2[o] * Z[n]
        # 역전파 2
        for k in range(64):
            for j in range(5):
                W1[k][j] = W1[k][j] + eta * delta1[j] * (X[i]).flatten()[k]

        # 출력 유니트
        # print(type(O[i]))
        # for k in range(10):
        #     O[i] = float(O[i])
        #     pos = O.index(min(O))
        #     if pos == 0:
        #         str1 = "ㄱ"
        #     elif pos == 1:
        #         str1 = "ㄴ"
        #     elif pos == 2:
        #         str1 = "ㄷ"
        #     elif pos == 3:
        #         str1 = "ㄹ"
        #     elif pos == 4:
        #         str1 = "ㅁ"
        #     elif pos == 5:
        #         str1 = "ㅂ"
        #     elif pos == 6:
        #         str1 = "ㅅ"
        #     elif pos == 7:
        #         str1 = "ㅇ"
        #     elif pos == 8:
        #         str1 = "ㅈ"
        #     elif pos == 9:
        #         str1 = "ㅋ"
        # print(str(O[i]) + " " + str1)
        # # print(O)

    print("")
    epoch += 1

noise_pattern = np.array([[1, 1, 0, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 0],
                          [0, 0, 1, 0, 0, 0, 1, 1],
                          [0, 0, 0, 0, 0, 0, 1, 1],
                          [0, 0, 0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 1],
                          [0, 0, 0, 0, 0, 0, 1, 1],
                          [0, 0, 0, 0, 0, 0, 1, 1]])

noise_pattern = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1]])

# 학습된 가중치를 기반으로 글자 분류
for k in range(5):
    A[k] = np.dot(noise_pattern.flatten(), W1[:, k])
    Z[k] = sigmoid1.forward(A[k])
for j in range(10):
    O[j] = np.dot(np.array(Z).flatten(), W2[:, j])
    O[j] = sigmoid2.forward(O[j])

# 출력 유니트
for k in range(10):
    O[i] = float(O[i])
    pos = O.index(min(O))
    if pos == 0:
        str1 = "ㄱ"
    elif pos == 1:
        str1 = "ㄴ"
    elif pos == 2:
        str1 = "ㄷ"
    elif pos == 3:
        str1 = "ㄹ"
    elif pos == 4:
        str1 = "ㅁ"
    elif pos == 5:
        str1 = "ㅂ"
    elif pos == 6:
        str1 = "ㅅ"
    elif pos == 7:
        str1 = "ㅇ"
    elif pos == 8:
        str1 = "ㅈ"
    elif pos == 9:
        str1 = "ㅋ"
    else:
        print("error")
print(str(O[pos]) + " " + str1)
