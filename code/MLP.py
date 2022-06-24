import numpy as np

from active_function import *
from graphics import *
from init_parameters import *


class MLP:

    #   x = list of features and observations
    #   y = list of targets and observations
    #   h_layers = list of numbers of hidden neurons
    #   max_epoch = max number iteration dataset
    #   l_rate = learning rate
    #   batch_size = dimension of mini batch - in this problem is always 1
    #   af = active function used
    def __init__(self, x, y, h_layers, max_epoch=1, l_rate=0.0001, batch_size=1, af="Sigmoid"):
        self.x = np.transpose(x)
        self.y = np.transpose(y)
        self.n_features = self.x.shape[0]
        self.n_targets = self.y.shape[0]
        self.layers = [self.n_features] + h_layers + [self.n_targets]
        self.max_epoch = max_epoch
        self.l_rate = l_rate
        self.batch_size = batch_size
        self.af = af
        # generate random weights in uniform distribution or with random values
        self.w, self.b = glorot_parameters(self.layers)

    def get_mini_batches(self, minibatch):
        return self.x[:, minibatch].reshape(self.n_features, self.batch_size), \
               self.y[:, minibatch].reshape(self.n_targets, self.batch_size)

    def get_shuffled_indexes(self):
        batches = np.arange(0, self.x.shape[1])
        np.random.shuffle(batches)
        return batches

    def backpropagation(self, val_x, val_y):
        loss = {}
        val = []
        n_val = len(val_x)
        val_xx, val_yy = np.transpose(val_x), np.transpose(val_y)
        for epoch in range(self.max_epoch):
            loss[epoch] = []
            # Stochastic Gradient Descent minibatch
            batches = self.get_shuffled_indexes()
            for minibatch in batches:
                x, y = self.get_mini_batches(minibatch)
                z = self.forward(x)
                loss[epoch].append(self.multiclass_cross_entropy(z[-1], y))
                self.backward(z, y)
            val.append(self.prediction(val_xx, val_yy))
        loss = [np.mean(loss[i]) for i in range(self.max_epoch)]
        return loss, val

    def forward(self, x):
        k = len(self.w)
        z = [x]
        for i in range(1, k):
            if self.af == "Sigmoid":
                z.append(Sigmoid(np.dot(self.w[i], z[i - 1]) + self.b[i]))
            elif self.af == "Relu":
                z.append(Relu(np.dot(self.w[i], z[i - 1]) + self.b[i]))
            elif self.af == "Tanh":
                z.append(Tanh(np.dot(self.w[i], z[i - 1]) + self.b[i]))
        z.append(Softmax(np.dot(self.w[k], z[k - 1]) + self.b[k]))
        return z

    def backward(self, a, y):
        gw, gb = self.gradients(a, y)
        # Update parameters
        for i in range(1, len(self.w) + 1):
            self.w[i] -= self.l_rate * gw[i]
            self.b[i] -= self.l_rate * gb[i]

    def gradients(self, z, y):
        dw, db = {}, {}
        n = len(self.w)
        m = 1 / self.batch_size
        dz = z[-1] - y
        for i in range(n, 0, -1):
            dw[i] = m * np.dot(dz, np.transpose(z[i - 1]))
            db[i] = m * np.sum(dz, axis=1, keepdims=True)
            if self.af == "Sigmoid":
                dz = np.dot(np.transpose(self.w[i]), dz) * Sigmoid(z[i - 1], True)
            elif self.af == "Relu":
                dz = np.dot(np.transpose(self.w[i]), dz) * Relu(z[i - 1], True)
            elif self.af == "Tanh":
                dz = np.dot(np.transpose(self.w[i]), dz) * Tanh(z[i - 1], True)
        return dw, db

    def multiclass_cross_entropy(self, z, y):
        return -(1 / self.batch_size) * np.sum(y * np.log(z))

    def prediction(self, x, y):
        return self.multiclass_cross_entropy(self.forward(x)[-1], y)

