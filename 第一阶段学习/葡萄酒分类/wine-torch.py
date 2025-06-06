
import torch
import numpy as np
import pandas as pd

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split


class Wine(Dataset):
    def __init__(self, inputs, labels):
        super().__init__()
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        if idx >= len(self.inputs):
            print("Out of Range.")
        return self.inputs[idx], self.labels[idx]

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear1 = nn.Linear(in_features, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, out_features)

        self.activation = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, labels, train_mode=True):
        output = self.linear1(inputs)
        output = self.activation(output)
        output = self.linear2(output)
        output = self.activation(output)
        output = self.linear3(output)
        output = self.activation(output)

        loss = self.loss_fn(output, labels)
        if train_mode:
            return loss
        else:
            predicts = torch.softmax(output, dim=1)
            return predicts, loss


def run_valid():
    correct_nums = 0
    sum_loss = 0
    for batch_inputs, batch_labels in validloader:

        predicts, loss = model(batch_inputs, batch_labels, train_mode=False)

        sum_loss += loss
        correct_nums += (predicts.argmax(dim=1) == batch_labels).sum()

    acc = correct_nums / len(validloader.dataset)
    return acc, sum_loss / len(validloader)

if __name__ == "__main__":
    raw_data = pd.read_csv("data/19-WineQT.csv")

    raw_labels = raw_data["quality"].to_numpy()
    map_labels = raw_labels - 3
    # map_labels = torch.tensor(map_labels, dtype=torch.long)
    classes = len(set(raw_labels))

    raw_inputs = raw_data.iloc[:, :-2]
    raw_inputs = ((raw_inputs - np.mean(raw_inputs, axis=0))/np.std(raw_inputs, axis=0)).to_numpy()
    raw_inputs = torch.tensor(raw_inputs, dtype=torch.float32)

    train_ratio = 0.9
    train_size  = int(len(raw_inputs)*train_ratio)
    valid_size  = len(raw_inputs) - train_size

    # train_inputs = raw_inputs[:train_size]
    # train_labels = map_labels[:train_size]

    # valid_inputs = raw_inputs[train_size:]
    # valid_labels = map_labels[train_size:]


    # trainset = Wine(train_inputs, train_labels)
    # validset = Wine(valid_inputs, valid_labels)

    dataset = Wine(raw_inputs, map_labels)

    trainset, validset = random_split(dataset, (train_size, valid_size))

    batch_size = 16
    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    validloader = DataLoader(validset, 8, shuffle=True)

    in_features  = raw_inputs.shape[1]
    out_features = classes

    model = Model(in_features, out_features)
    lr = 0.001

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 100

    for epoch in range(epochs):
        for batch_inputs, batch_labels in trainloader:

            loss = model(batch_inputs, batch_labels)
            loss.backward()

            optim.step()
            optim.zero_grad()

        if epoch % 10 == 0:
            acc, valid_loss = run_valid()
            print("Epoch: {}, Loss: {}, Valid Loss: {}, Acc: {:.2f}%".format(epoch, loss, valid_loss, acc*100))
    pass