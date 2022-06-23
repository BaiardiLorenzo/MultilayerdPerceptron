import math


# GLOROT value d for the uniform distribution
import numpy as np


def Glorot(parents, children):
    return math.sqrt(6 / (parents + children))


# init weights with glorot - Uniform(-d, d)
def glorot_weights(l):
    w, b = {}, {}
    n = len(l)
    for i in range(1, n - 1):
        d = Glorot(l[i - 1], l[i + 1])
        w[i] = np.random.uniform(-d, d, (l[i], l[i - 1]))
        b[i] = np.random.uniform(-d, d, (l[i], 1))
    d = Glorot(l[n - 1], 0)
    w[n - 1] = np.random.uniform(-d, d, (l[n - 1], l[n - 2]))
    b[n - 1] = np.random.uniform(-d, d, (l[n - 1], 1))
    return w, b


# init weights with random values
def random_weights(layers):
    w, b = {}, {}
    n = len(layers)
    for i in range(1, n):
        w[i] = np.random.randn(layers[i], layers[i - 1])
        b[i] = np.random.randn(layers[i], 1)
    return w, b
