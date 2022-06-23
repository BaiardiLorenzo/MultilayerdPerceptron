import pandas as pd
from MLP import MLP
from graphics import *


def draw_nn(l):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, l)
    plt.savefig("./documentation/images/neural_network.jpg")


def main():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv", index_col=False)
    x = df.iloc[:9000, 3:8].to_numpy()
    y = df.iloc[:9000, 10:14].to_numpy()
    predict_x = df.iloc[9000:10000, 3:8].to_numpy()
    h_layers = [4]
    mlp = MLP(x, y, h_layers, max_epoch=50)
    # draw_nn(mlp.layers)

    e = mlp.backpropagation()


if __name__ == '__main__':
    main()
