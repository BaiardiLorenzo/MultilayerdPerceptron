import numpy as np


# ---- ACTIVE FUNCTIONS ----

# SIGMOID
def Sigmoid(x, derivation=False):
    if derivation:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


# SOFTMAX - Use for Classification
def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# RELU
def Relu(x, derivation=False):
    if derivation:
        return np.array(x > 0, dtype=float)
    else:
        return np.maximum(x, 0)


# TANH
def Tanh(x, derivation=False):
    if derivation:
        return 1 - np.power(np.tanh(x), 2)
    else:
        return np.tanh(x)

