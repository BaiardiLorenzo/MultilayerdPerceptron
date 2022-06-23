import numpy as np


# ---- ACTIVE FUNCTIONS ----

# SIGMOID
def Sigmoid(x, derivation=False):
    if derivation:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


# RELU
def Relu(x, derivation=False):
    if derivation:
        return np.array(x >= 0).astype(int)
    else:
        return np.maximum(0, x)


# TANH
def Tanh(x, derivation=False):
    if derivation:
        return 1 - np.power(np.tanh(x), 2)
    else:
        return np.tanh(x)
