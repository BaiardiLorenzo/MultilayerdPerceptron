import numpy as np


# MULTICLASS - SOFTMAX
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
