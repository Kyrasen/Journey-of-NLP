import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

# 继承torch里面的Dataset类
class Wine(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.size = len(labels)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if idx > self.size:
            print("Out of range")
        return self.inputs[idx], self.labels[idx]
    
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear1 = nn.Linear(in_features, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, out_features)

        self.activation = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss()包含了Onehot处理

    def forward(self, inputs, labels, train_mode=True):
        output = self.linear1(inputs)
        output = self.activation(output)
        output = self.linear1(output)
        output = self.activation(output)
        output = self.linear1(output)
        output = self.activation(output)

        loss = self.loss_fn.forward(output, labels)
        if train_mode:
            return loss
        else:
            predicts = torch.softmax(output, dim=1)
            return predicts, loss


def run_valid():
    current_nums = 0
    sum_loss = 0
    for batch_inputs, batch_labels in validloader:
        predicts, loss = model(batch_inputs, batch_labels, train_mode=False)
        sum_loss += loss
        correct_nums += (predicts.argmax(dim=1) == batch_labels).sum()
    acc = correct_nums / len(validloader.dataset)
    return acc, sum_loss/len(validloader)



if __name__ == "__main__":
    raw_data = pd.read_csv("data/19-WineQT.csv")
    raw_labels = raw_data["quality"].to_numpy()
    map_labels = raw_labels - 3
    classes = len(set(map_labels))


    raw_inputs = raw_data.iloc[:, :-2].to_numpy()
    raw_inputs = (raw_inputs - np.mean(raw_inputs, axis = 0) )/ np.std(raw_inputs, axis = 0)
    raw_inputs = torch.tensor(raw_inputs, dtype = torch.float32)

    train_ratio = 0.9
    train_size = int(train_ratio * len(raw_inputs))
    valid_size = len(raw_inputs) - train_size

    # train_inputs = raw_inputs[:train_size]
    # train_labels = map_labels[:train_size]

    # valid_inputs = raw_inputs[train_size:]
    # valid_labels = map_labels[train_size:]

    # trainset = Dataset(train_inputs, train_labels)
    # validset = Dataset(valid_inputs, valid_labels)

    dataset = Wine(raw_inputs, map_labels)

    trainset, validset = random_split(dataset, (train_size, valid_size))  #随机打乱顺序并切分训练集和测试集

    batch_size = 8
    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    validloader = DataLoader(validset, 16, shuffle = True)

    in_features = raw_inputs.shape[1] # 有多少个特征
    out_features = classes
    
    model = Model(in_features, out_features)

    epochs = 100
    lr = 0.01

    optim = torch.optim.Adam(model.parameters(), lr = lr)

    for e in range(epochs):
        for batch_inputs, batch_labels in trainloader:
            loss = model(batch_inputs, batch_labels)
            loss.backward()

            optim.step()
            optim.zero_grad()

        if e % 10 == 0:
            acc, valid_loss = run_valid()
            print(f"Epoch: {e}, Loss:{loss}, valid loss:{valid_loss}, acc:{acc}")

    pass
    