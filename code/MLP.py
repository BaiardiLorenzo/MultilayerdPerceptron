from matplotlib import pyplot as plt

from active_function import *
import numpy as np

from init_weights import *


class MLP:
    #   x = list of inputs
    #   y = list of outputs to predict
    #   h_layers = list of numbers of hidden neurons
    #   max_epoch = max number of epoch
    #   l_rate = learning rate of MLP
    #   batch_size = dimension of batch size for mini batch
    def __init__(self, x, y, h_layers, max_epoch=1, l_rate=0.5, batch_size=64, af="Relu"):
        self.x = np.transpose(x)
        self.y = y
        self.layers = [len(self.x)] + h_layers + [len(self.y[0])]
        self.max_epoch = max_epoch
        self.l_rate = l_rate
        self.batch_size = batch_size
        self.af = af
        # generate random weights in uniform distribution
        self.w, self.b = random_weights(self.layers)

    # backpropagation - to train mlp
    def backpropagation(self):
        c = []
        for i in range(self.max_epoch):
            # add stochastic minibatch passing data
            # batches = np.arange(0, self.x.shape[1])
            # np.random.shuffle(batches)
            # batches = np.split(batches, self.batch_size)
            # for minibatch in batches:
                # n_x, n_y = self.x.shape[0], self.y.shape[1]
                # x, y = self.x[:, minibatch].reshape(n_x, 1), self.y[minibatch, :].reshape(1, n_y)
            x, y = self.x, self.y
            z = self.forward_propagation(x)
            cost = self.cross_entropy(z[-1], np.transpose(y))
            c.append(cost)
            self.backward_propagation(z, y)
            #print("cost after", i, " iterations is ", cost)

        plt_cross = plt.figure(1)
        plt.title = "Loss"
        plt.plot(np.arange(0, self.max_epoch), c)
        plt.xlabel("iterations")
        plt.ylabel("cross-entropy-loss")
        plt.show()

    # forward propagation
    def forward_propagation(self, x):
        k = len(self.w)
        z = [np.array(x)]
        for i in range(1, k):
            if self.af == "Relu":
                z.append(Relu(np.dot(self.w[i], z[i - 1]) + self.b[i]))
            elif self.af == "Sigmoid":
                z.append(Sigmoid(np.dot(self.w[i], z[i - 1]) + self.b[i]))
            elif self.af == "Tanh":
                z.append(Tanh(np.dot(self.w[i], z[i - 1]) + self.b[i]))
        # multiclass - softmax
        z.append(self.softmax(np.dot(self.w[k], z[k - 1]) + self.b[k]))
        return z

    # MULTICLASS - SOFTMAX
    def softmax(self, x):
        e = np.exp(x/100)
        return e / np.sum(e)

    def cross_entropy(self, z, y):
        return -(1 / (y.shape[1])) * np.sum(y * np.log(z))

    # calculate the gradients
    def grad(self, z, y):
        dw, db = {}, {}
        n = len(self.w)
        m = 1 / y.shape[0]
        dz = z[-1] - np.transpose(y)
        for i in range(n, 0, -1):
            dw[i] = m * np.dot(dz, np.transpose(z[i - 1]))
            db[i] = m * np.sum(dz, axis=1, keepdims=True)
            if self.af == "Relu":
                dz = np.dot(np.transpose(self.w[i]), dz) * Relu(z[i - 1], True)
            elif self.af == "Sigmoid":
                dz = np.dot(np.transpose(self.w[i]), dz) * Sigmoid(z[i - 1], True)
            elif self.af == "Tanh":
                dz = np.dot(np.transpose(self.w[i]), dz) * Tanh(z[i - 1], True)
        return dw, db

    # backward propagation
    def backward_propagation(self, a, y):
        gw, gb = self.grad(a, y)
        for i in range(1, len(self.w)+1):
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
