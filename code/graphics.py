import numpy as np
from matplotlib import pyplot as plt


def PlotLoss(loss, i):
    plt_loss = plt.figure(10+i)
    plt.title("Error training: Dataset: "+str(i))
    plt.plot(np.arange(0, len(loss)), loss, color="firebrick")
    plt.xlabel("Epoch")
    plt.ylabel("Multiclass_Cross_Entropy")
    plt.savefig("./documentation/images/loss"+str(i)+".jpg")


def PlotValidation(val, i):
    plt_val = plt.figure(20+i)
    plt.title("Validation: Dataset: "+str(i))
    plt.plot(np.arange(0, len(val)), val, color="midnightblue")
    plt.xlabel("Epoch")
    plt.ylabel("Multiclass_Cross_Entropy")
    plt.savefig("./documentation/images/validation"+str(i)+".jpg")


# REFERENCES - https://gist.github.com/craffel/2d727968c3aaebd10359
# Draw a neural network in plot
def PlotNetwork(l):
    plt.style.use('default')
    plt_net = plt.figure(10)
    plt.title("Esempio di Rete")
    ax = plt_net.gca()
    ax.axis('off')
    draw_neural_net(ax, l)
    plt_net.savefig("./documentation/images/neural_network.jpg")


def draw_neural_net(ax, layer_sizes, left=.1, right=.9, bottom=.1, top=.9):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)
