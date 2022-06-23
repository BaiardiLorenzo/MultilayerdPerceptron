import warnings

import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from MLP import MLP
from graphics import *


def MachineFailureDataset():
    # Dataset from url: https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
    # Tell us if the machine have a particular failure with his features
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv", index_col=False)
    train_x = df.iloc[:9000, 3:8].to_numpy()
    train_y = df.iloc[:9000, 10:14].to_numpy()
    val_x = df.iloc[9000:, 3:8].to_numpy()
    val_y = df.iloc[9000:, 10:14].to_numpy()
    return train_x, train_y, val_x, val_y


def FrogDataset():
    # Download dataset from url: https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29
    # write the file directory on the methods pd.read_csv
    # Find the correct family for this frog with this features
    df = pd.read_csv("./es_dataset/Frogs_MFCCs.csv")
    x, y = df.iloc[:, :21].to_numpy(), LabelBinarizer().fit_transform(df.iloc[:, 24].to_numpy())
    index = np.arange(0, len(x))
    np.random.shuffle(index)
    x, y = x[index, :], y[index, :]
    train_x, train_y = x[:6000, :], y[:6000, :]
    val_x, val_y = x[6000:, :], y[6000:, :]
    return train_x, train_y, val_x, val_y


def main():
    warnings.filterwarnings('ignore')
    # MachineFailureDataset
    train_x, train_y, val_x, val_y = MachineFailureDataset()
    h_layers = [4]
    nn1 = MLP(train_x, train_y, h_layers, max_epoch=50)
    loss = nn1.backpropagation()
    # validation = nn1.prediction(val_x, val_y)
    PlotLoss(loss, 1)
    # PlotValidation(validation, 1)

    # FrogDataset
    train_x, train_y, val_x, val_y = FrogDataset()
    h_layers = [7, 6]
    nn2 = MLP(train_x, train_y, h_layers, max_epoch=50)
    loss = nn2.backpropagation()
    # validation = nn2.prediction(val_x, val_y)
    PlotLoss(loss, 2)
    # PlotValidation(validation, 2)


if __name__ == '__main__':
    main()
