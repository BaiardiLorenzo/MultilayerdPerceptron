from matplotlib import pyplot as plt

from active_function import *
import numpy as np

from init_weights import glorot_weights
from softmax import softmax


class MLP:
    #   x = list of inputs
    #   y = list of outputs to predict
    #   h_layers = list of numbers of hidden neurons
    #   max_epoch = max number of epoch
    #   l_rate = learning rate of MLP
    #   batch_size = dimension of batch size for mini batch
    def __init__(self, x, y, h_layers, max_epoch=1, l_rate=0.1, batch_size=32, af="Relu"):
        self.x = np.transpose(x)
        self.y = y
        self.layers = [len(self.x)] + h_layers + [len(self.y[0])]
        self.max_epoch = max_epoch
        self.l_rate = l_rate
        self.batch_size = batch_size
        self.af = af
        # generate random weights in uniform distribution
        self.w, self.b = glorot_weights(self.layers)

    # backpropagation - to train mlp
    def backpropagation(self):
        for i in range(self.max_epoch):
            a = self.forward_propagation(self.x)
            self.backward_propagation(a)

    # forward propagation
    def forward_propagation(self, x):
        a = [x]
        k = len(self.w) - 1
        for j in range(1, k):
            if self.af == "Relu":
                a.append(Relu(np.dot(self.w[j], a[j - 1]) + self.b[j]))
            elif self.af == "Sigmoid":
                a.append(Sigmoid(np.dot(self.w[j], a[j - 1]) + self.b[j]))
        # multiclass - softmax
        a.append(softmax(np.dot(self.w[k], a[k - 1]) + self.b[k]))
        return a

    # stochastic gradient descent
    def sgd(self, a):
        dw, db = {}, {}
        j = len(self.w) - 1
        # number of size output
        m = 1 / self.layers[-1]
        # delta output of j - first delta on the top
        dz = np.transpose(self.y) - a[-1]
        for i in range(j, 0, -1):
            dw[i] = m * np.dot(dz, np.transpose(a[i - 1]))
            db[i] = m * np.sum(dz)
            # derivation active function
            if self.af == "Relu":
                dz = np.dot(np.transpose(self.w[i]), dz) * Relu(a[i - 1], True)
            elif self.af == "Sigmoid":
                dz = np.dot(np.transpose(self.w[i]), dz) * Sigmoid(a[i - 1], True)
        return dw, db

    # backward propagation
    def backward_propagation(self, a):
        # gradients
        gw, gb = self.sgd(a)
        for i in range(1, len(self.w)):
            self.w[i] -= self.l_rate * gw[i]
            self.b[i] -= self.l_rate * gb[i]

    # adjustment probability
    def adj(self, a):
        for i in range(len(a[0])):
            if a[0, i] > 0.5:
                a[0, i] = 1
            else:
                a[0, i] = 0
        return a

    def mce(self, a, y):
        return

    # prediction value
    def prediction(self, x):
        return self.adj(self.forward_propagation(x)[self.layers[-1]])

    def plot_errors(self, e):
        plot_times = plt.figure(1)
        plt.title("Error training")
        plt.plot(range(0, self.max_epoch), e, color="midnightblue")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.savefig("./documentation/graphics/errors.jpg")
