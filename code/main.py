import pandas as pd
from matplotlib import pyplot as plt
from MLP import MLP
from graphics import *


def draw_nn(l):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, l)
    fig.savefig('nn.png')


def main():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv", index_col=False)
    x = df.iloc[:9000, 3:8].to_numpy()
    y = df.iloc[:9000, 10:14].to_numpy()
    predict_x = df.iloc[9000:10000, 3:8].to_numpy()
    h_layers = [4, 5, 8]
    mlp = MLP(x, y, h_layers, max_epoch=1000)
    draw_nn(mlp.layers)

    e = mlp.backpropagation()
    # nn.plot_errors(e)




if __name__ == '__main__':
    main()
