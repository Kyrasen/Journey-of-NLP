import pandas as pd
import numpy as np


'''
单特征 函数
'''
def f(x, w):
    return x * w

def loss_func(y_pred, label):
    return (y_pred - label)**2

if __name__ == "__main__":

    # 数据处理
    raw_data = pd.read_csv("boston_house_prices.csv").drop("CHAS", axis=1)
    raw_inputs = raw_data["CRIM"].to_numpy().reshape(-1, 1)
    raw_labels = raw_data["MEDV"].to_numpy().reshape(-1, 1)

    # 归一化  
    inputs = (raw_inputs - np.mean(raw_inputs)) / np.std(raw_inputs)
    labels = (raw_labels - np.mean(raw_labels)) / np.std(raw_labels)

    # 训练数据
    epoches = 100
    w = 0.
    lr = 0.001
    for epoch in range(epoches):
        for x, label in zip(inputs, labels):
            y_pred = f(x, w)
            loss = loss_func(y_pred, label)
            G = 2 * (y_pred - label)
            dw = G * x
            w = w - lr * dw
        print(f"loss: {loss}")
        print(f"epoch: {epoch}")

    pass