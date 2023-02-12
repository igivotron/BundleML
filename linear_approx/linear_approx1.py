import numpy as np
import matplotlib.pyplot as plt


### Paramètres de la génération de données ###
n = 100
m = 6
p = 2
a = 150

### Paramètres de l'IA ###
it = 10000
lr = 0.01

### Générateur de données ###
X = np.arange(n)
Y = np.array([m*i + p+ np.random.randn() *a for i in range(n)])

### IA ###
def predict(w, q):
    return X * w + q

def loss(w, q):
    return np.average(pow(predict(w, q) - Y, 2))

def train(it, lr):
    w = 0
    q = 0
    for i in range(it):
        current_loss = loss(w, q)
        if loss(w + lr, q + lr) < current_loss:
            w += lr
            q += lr
        elif loss(w + lr, q - lr) < current_loss:
            w += lr
            q -= lr
        elif loss(w, q + lr) < current_loss:
            q += lr
        elif loss(w, q - lr) < current_loss:
            q -= lr
        
        else:
            return w,q

    raise Exception("N'a pas su converger") 

w, q= train(it, lr)
print(w,q)

### Plot des ressources ###
plt.scatter(X,Y)
plt.plot(X,predict(w,q))
plt.show()