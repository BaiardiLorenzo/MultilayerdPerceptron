from matplotlib import pyplot as plt

from active_function import *
import numpy as np

from init_weights import *


class MLP:

    #   x = list of features and observations
    #   y = list of targets and observations
    #   h_layers = list of numbers of hidden neurons
    #   max_epoch = max number iteration dataset
    #   l_rate = learning rate
    #   batch_size = dimension of mini batch
    def __init__(self, x, y, h_layers, max_epoch=1, l_rate=0.0001, batch_size=1):
        features = x.shape[1]
        targets = y.shape[1]
        self.layers = [features] + h_layers + [targets]
        self.x = np.transpose(x)
        self.y = np.transpose(y)
        self.max_epoch = max_epoch
        self.l_rate = l_rate
        self.batch_size = batch_size
        self.w, self.b = glorot_weights(self.layers)  # generate random weights in uniform distribution

    def backpropagation(self):
        loss, val = {}, {}
        for epoch in range(self.max_epoch):
            # add stochastic minibatch passing data
            loss[epoch] = []
            batches = np.arange(0, self.x.shape[1])
            np.random.shuffle(batches)
            for minibatch in batches:
                n_x, n_y = self.x.shape[0], self.y.shape[0]
                x, y = self.x[:, minibatch].reshape(n_x, self.batch_size), self.y[:, minibatch].reshape(n_y, self.batch_size)
                z = self.forward_propagation(x)
                loss[epoch].append(self.cross_entropy(z[-1], y))
                self.backward_propagation(z, y)
                # print("cost after", minibatch, " iterations is ", cost)
        loss = [np.mean(loss[i]) for i in range(self.max_epoch)]
        plt_cross = plt.figure(1)
        plt.title = "Loss"
        plt.plot(np.arange(0, self.max_epoch), loss)
        plt.xlabel("iterations")
        plt.ylabel("cross-entropy-loss")
        plt.show()

    def forward_propagation(self, x):
        k = len(self.w)
        z = [x]
        for i in range(1, k):
            z.append(Sigmoid(np.dot(self.w[i], z[i - 1]) + self.b[i]))
        # multiclass - softmax
        z.append(self.softmax(np.dot(self.w[k], z[k - 1]) + self.b[k]))
        return z

    # calculate the gradients
    def grad(self, z, y):
        dw, db = {}, {}
        n = len(self.w)
        m = 1 / self.batch_size
        dz = z[-1] - y
        for i in range(n, 0, -1):
            dw[i] = m * np.dot(dz, np.transpose(z[i - 1]))
            db[i] = m * np.sum(dz, axis=1, keepdims=True)
            dz = np.dot(np.transpose(self.w[i]), dz) * Sigmoid(z[i - 1], True)
        return dw, db

    def backward_propagation(self, a, y):
        gw, gb = self.grad(a, y)
        for i in range(1, len(self.w)+1):
            self.w[i] -= self.l_rate * gw[i]
            self.b[i] -= self.l_rate * gb[i]

    def softmax(self, x):     # MULTICLASS - SOFTMAX
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def cross_entropy(self, z, y):
        return -(1 / self.batch_size) * np.sum(y * np.log(z))



    def prediction(self, x):
        return self.forward_propagation(x)[self.layers[-1]]

