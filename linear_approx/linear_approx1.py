import numpy as np
import matplotlib.pyplot as plt

n = 100
m = 7.5
a = 100

it = 10000
lr = 0.01

X = np.arange(n)
Y = np.array([m*i + np.random.randn() *a for i in range(n)])


def predict(w):
    return X * w

def loss(w):
    return np.average(pow(predict(w) - Y, 2))

def train(it, lr):
    w = 0
    for i in range(it):
        current_loss = loss(w)
        if loss(w + lr) < current_loss:
            w += lr
        elif loss(w - lr) < current_loss:
            w -= lr
        else:
            return w

    raise Exception("N'a pas su converger") 

w = train(it, lr)
print(w)
plt.scatter(X,Y)
plt.plot(X,predict(w))
plt.show()