import math

import numpy as np


# MULTICLASS - SOFTMAX
def soft_max(x):
    return np.e(x) / np.sum(np.e(x))


# SIGMOID
def sigmoid(x, derivation=False):
    if derivation:
        return
    else:
        return 1 / (1 + math.exp(-x))


# STEP FUNCTION - IDEAL
def Relu(x, derivation=False):
    if derivation:
        if x >= 0:
            return 1
        else:
            return 0
    else:
        return max(0, x)


# REGULAR IMPLEMENTATION OF RELU
def leaky_Relu(x, derivation=False):
    if derivation:
        if x >= 0:
            return 1
        else:
            return 0.01
    else:
        if x >= 0:
            return x
        else:
            return 0.01 * x


def Glorot(f, c):
    return 6 / (math.sqrt(f + c))

