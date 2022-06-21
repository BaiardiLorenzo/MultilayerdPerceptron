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
