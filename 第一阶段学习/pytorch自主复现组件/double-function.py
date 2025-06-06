import pandas as pd
import numpy as np

'''
    多特征
'''

def fun(x1, x2, w1, w2):
    return w1*x1 + x2*w2

def loss_func(y_pred, label):
    return (y_pred - label)**2

if __name__ == "__main__":
    raw_data = pd.read_csv("day3/boston_house_prices.csv").drop("CHAS", axis=1)
    raw_age = raw_data["AGE"].to_numpy()
    raw_dis = raw_data["DIS"].to_numpy()

    raw_labels = raw_data["MEDV"].to_numpy()

    # 归一化
    inputs_age = (raw_age - np.mean(raw_age)) / np.std(raw_age)
    inputs_dis = (raw_dis - np.mean(raw_dis)) / np.std(raw_dis)
    inputs_labels = (raw_labels -np.mean(raw_labels)) / np.std(raw_labels)

    # 训练数据
    epochs = 100
    lr = 0.001
    w1 = 0.
    w2 = 0.
    for epoch in range(epochs):
        for age, dis, label in zip(inputs_age, inputs_dis, inputs_labels):
            y_pred = fun(age, dis, w1, w2)
            loss = loss_func(y_pred, label)
            G = 2 * (y_pred - label) 
            dw1 = G * age
            dw2 = G * dis
            w1 = w1 - lr * dw1
            w2 = w2 - lr * dw2
        print(f"loss : {loss}")   

    pass

                                                  