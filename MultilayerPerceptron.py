from functions import *


class MLP:

    #   x = list of inputs
    #   y = list of outputs to predict
    #   h_layers = list of numbers of hidden neurons
    #   max_epoch = max number of epoch
    #   l_rate = learning rate of MLP
    #   b_size = batch size for minibatch
    def __init__(self, x, y, h_layers, max_epoch=1, l_rate=0.1, b_size=32):
        self.x = x
        self.y = y
        self.layers = len(x) + h_layers + len(y)
        self.max_epoch = max_epoch
        self.l_rate = l_rate
        self.b_size = b_size
        self.w = {}
        self.b = {}
        # generate random weights
        self.init_weight_random()
        # generate random weights in uniform distribution
        # self.init_weights_glorot()

    # init weights with glorot
    def init_weights_glorot(self):
        for i in range(1, len(self.layers)):
            d = Glorot(self.layers[i - 1], self.layers[i])
            self.w[i] = np.random.uniform(-d, d, [self.layers[i], self.layers[i - 1]])
            self.b[i] = np.random.uniform(-d, d, [self.layers[i], 1])

    # init weights with random values
    def init_weight_random(self):
        for i in range(1, len(self.layers)):
            self.w[i] = np.random.randn(self.layers[i], self.layers[i - 1])
            self.b[i] = np.random.randn(self.layers[i], 1)

    # forward propagation
    def forward_propagation(self, x):
        act = x
        k = len(self.w)
        for j in range(1, k):
            aj = np.dot(self.w[j], act[j - 1]) + self.b[j]
            act[j] = leaky_Relu(aj)  # or use sigmoid - ideal Relu
        aj = np.dot(self.w[k], act[k - 1]) + self.b[k]
        act[k] = soft_max(aj)  # multiclass - softmax
        return act

    # stochastic gradient descent
    def sgd(self, act):
        dw, db = {}, {}
        j = len(self.w)
        m = 1 / len(self.y)  # number of size output
        dz = act[j] - self.y  # delta output of j - first delta on the top
        for i in reversed(range(1, j + 1)):
            dw[i] = m * np.dot(dz, act[i-1])
            db[i] = m * np.sum(dz)
            dz = np.dot(self.w[i], dz) * leaky_Relu(act[i-1], True)  # derivation active function
        return dw, db

    # backward propagation
    def backward_propagation(self, act):
        # gradients
        gw, gb = self.sgd(act)
        for i in range(1, len(self.w)):
            self.w[i] -= self.l_rate * gw[i]
            self.b[i] -= self.l_rate * gb[i]

    # backpropagation - train the mlp
    def backpropagation(self):
        for i in range(self.max_epoch):
            self.backward_propagation(self.forward_propagation(self.x))
