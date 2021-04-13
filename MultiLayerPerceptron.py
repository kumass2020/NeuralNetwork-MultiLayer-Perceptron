import numpy as np
from math import ceil, floor


def init_network():
    # network = {}
    # network['']

    x1 = np.asfarray([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1]])
    d1 = np.asfarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    x2 = np.asfarray([[1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1]])
    d2 = np.asfarray([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    x3 = np.asfarray([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1]])
    d3 = np.asfarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    x4 = np.asfarray([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1]])
    d4 = np.asfarray([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    x5 = np.asfarray([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1]])
    d5 = np.asfarray([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    x6 = np.asfarray([[1, 1, 0, 0, 0, 0, 1, 1],
                  [1, 1, 0, 0, 0, 0, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 0, 0, 0, 0, 1, 1],
                  [1, 1, 0, 0, 0, 0, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1]])
    d6 = np.asfarray([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    x7 = np.asfarray([[0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1]])
    d7 = np.asfarray([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

    x8 = np.asfarray([[0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 0, 0, 1, 1, 0, 0, 0]])
    d8 = np.asfarray([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    x9 = np.asfarray([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [1, 1, 1, 0, 0, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0, 1, 1]])
    d9 = np.asfarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

    x10 = np.asfarray([[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1]])
    d10 = np.asfarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    # 가중치
    # W1 = np.full((64, 5), 0.5)
    # W2 = np.full((5, 10), 0.5)
    W1 = np.random.normal(scale=0.1, size=(64, 5))
    W2 = np.random.normal(scale=0.1, size=(5, 10))

    # # 은닉층 활성화 함수(Sigmoid) 전, 후
    # A = np.asfarray([0, 0, 0, 0, 0])
    # Z = np.asfarray([0, 0, 0, 0, 0])

    # X: 입력패턴, D: 출력패턴
    X = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    D = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]

    return X, D, W1, W2


def makeNoise(x):
    if x == 0:
        x = 1
    elif x == 1:
        x = 0
    return x


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        return out


# 입력층 : 64개
# 은닉층 : 5개
# 출력층 : 10개
sigmoid1 = Sigmoid()
sigmoid2 = Sigmoid()
offset = 0
momentum = 1.0
eta = 0.1
bias1 = [0.0 for i in range(5)]
bias2 = [0.0 for i in range(10)]
# bias1 = [0.5, 1, 1, 1, 1]
# bias2 = [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1]
X, D, W1, W2 = init_network()
epoch = 0

# 은닉층 오차
delta1 = np.full((10, 5), 0.0)
# 출력층 오차
delta2 = np.full((10, 10), 0.0)
A = [0.0 for i in range(5)]
Z = [0.0 for i in range(5)]
O = np.full((10, 10), 0.0)
tmp = np.full((64, 5), 0.0)

test = X[0]
test = test.flatten()

while epoch < 100001:
    print("epoch: " + str(epoch))
    for i in range(10):  # ㄱ, ㄴ, ㄷ, ...
        # print("i: " + str(i))
        for k in range(5):  # A[0], A[1], ...

            # test
            test4 = X[i].flatten()
            test5 = W1[:, k]

            tmp = X[i]
            # epoch 두 번마다 노이즈 넣어 데이터 셋 증가 효과
            if epoch % 2 == 0:
                tmp_x = np.random.randint(0, 8)
                tmp_y = np.random.randint(0, 8)
                tmp[tmp_x][tmp_y] = makeNoise(tmp[tmp_x][tmp_y])

            # 은닉층 업데이트
            A[k] = np.dot(tmp.flatten(), W1[:, k]) + bias1[k]

            # 은닉층 내에서 활성화 함수(시그모이드) 적용
            Z[k] = sigmoid1.forward(A[k])

        test3 = Z
        for j in range(10):
            # print("j: " + str(j))
            test1 = np.asfarray(Z).flatten()
            test2 = W2[:, j]

            # 출력층 업데이트
            O[i][j] = np.dot(np.asfarray(Z).flatten(), W2[:, j]) + bias2[j]

            # 출력층 내에서 활성화 함수(시그모이드) 적용
            O[i][j] = sigmoid2.forward(O[i][j])

            test6 = O[i][j]
            test7 = 1 - O[i][j]
            test8 = (D[i])[j] - O[i][j]

            # print("오차:", (D[i])[i] - O[i][j])
            delta2[i][j] = O[i][j] * (1 - O[i][j]) * ((D[i])[j] - O[i][j])
            # delta2 = D[i] - O[i]

        for m in range(5):
            # print("m: " + str(m))
            summ = 0
            for n in range(10):
                # print("n: " + str(n))
                summ += delta2[i][n] * W2[m][n]
            delta1[i][m] = Z[m] * (1 - Z[m]) * summ
        # delta1 = np.asfarray(Z) * (1.0 - np.asfarray(Z)) * np.dot(W2.T, delta2)

        # 역전파 1
        for n in range(5):
            for o in range(10):
                W2[n][o] = momentum * W2[n][o] + eta * delta2[i][o] * Z[n]

        # 역전파 2
        for k in range(64):
            for j in range(5):
                W1[k][j] = momentum * W1[k][j] + eta * delta1[i][j] * (X[i]).flatten()[k]

        print()
        print(str(O[i]) + " " + str(i))

    print("")
    epoch += 1

# noise_pattern = np.asfarray([[1, 1, 0, 1, 1, 1, 1, 1],
#                           [1, 1, 1, 1, 1, 1, 1, 0],
#                           [0, 0, 1, 0, 0, 0, 1, 1],
#                           [0, 0, 0, 0, 0, 0, 1, 1],
#                           [0, 0, 0, 0, 0, 1, 0, 1],
#                           [0, 0, 0, 0, 0, 0, 1, 1],
#                           [0, 0, 0, 0, 0, 0, 1, 1],
#                           [0, 0, 0, 0, 0, 0, 1, 1]])

noise_pattern = np.asfarray([[1, 1, 1, 1, 1, 1, 1, 1],
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
    O[0][j] = np.dot(np.asfarray(Z).flatten(), W2[:, j])
    O[0][j] = sigmoid2.forward(O[0][j])

# 출력 유니트
result = O[0].tolist()
for i in range(10):
    result[i] = 1 - result[i]
# for i in range(10):
#     result[i] = float(result[i])
pos = result.index(max(result))
if pos == 0:
    str1 = "ㄱ" + " - " + str(result[0])
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
print(str(result) + " " + str1)
