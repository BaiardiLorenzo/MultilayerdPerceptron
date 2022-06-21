import math


# GLOROT value d for the uniform distribution
import numpy as np


def Glorot(f, c):
    return math.sqrt(6 / (f + c))


# init weights with glorot - Uniform(-d, d)
def glorot_weights(l):
    w, b = [], []
    n = len(l)
    d = Glorot(0, l[1])
    w.append(np.random.uniform(-d, d, (l[0], 1)))
    b.append(np.random.uniform(-d, d, (l[0], 1)))
    for i in range(1, n - 1):
        d = Glorot(l[i - 1], l[i + 1])
        w.append(np.random.uniform(-d, d, (l[i], l[i - 1])))
        b.append(np.random.uniform(-d, d, (l[i], 1)))
    d = Glorot(l[n - 1], 0)
    w.append(np.random.uniform(-d, d, (l[n - 1], l[n - 2])))
    b.append(np.random.uniform(-d, d, (l[n - 1], 1)))
    return w, b


# init weights with random values
def random_weights(layers):
    w, b = [], []
    w.append(np.random.randn(layers[0], 1))
    w.append(np.random.randn(layers[0], 1))
    for i in range(1, len(layers)):
        w.append(np.random.randn(layers[i], layers[i - 1]))
        b.append(np.random.randn(layers[i]))
    return w, b
