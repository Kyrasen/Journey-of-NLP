import numpy as np

'''
    用方差来做损失函数
'''

# 平均值
def mae(labels, predicts):
    return (np.sum(np.abs(labels - predicts)))/ len(labels)

# 方差 S
def mse(labels, predicts):
    sum = 0
    for i in range(len(labels)):
        sum = sum + np.abs(labels[i] - predicts[i])**2
    return (np.sum((labels-predicts)**2))/ len(labels)


if __name__ == '__main__':
    np.random.seed(14)
    labels = np.arange(10)
    predicts = np.arange(10) + np.random.randn(10)

    mse = mse(labels, predicts)
    mae = mae(labels, predicts)
    print(f"mse : {mse}")
    print(f"mae : {mae}")

