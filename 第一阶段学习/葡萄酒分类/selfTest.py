import pandas as pd
import numpy as np
from torch.utils.data import Dataset,random_split, DataLoader
import torch
import torch.nn as nn

class Wine(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
    
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, out_features)

        self.activation = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, inputs, labels, train_mode = True):
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
            predicts = torch.softmax(output, dim=1) # 这里会返回对应标签的匹配概率
            return predicts, loss

def run_valid():
    current_num = 0
    sum_loss = 0
    for batch_inputs, batch_labels in validloader:
        predicts, loss = model(batch_inputs, batch_labels, train_mode=False)
        current_num += (predicts.argmax(dim=1) == batch_labels).sum()
        sum_loss += loss
    
    return current_num/len(validloader.dataset), sum_loss/len(validloader)


if __name__ == "__main__":
    raw_data =pd.read_csv("data/19-WineQT.csv") 
    raw_labels = raw_data["quality"]
    map_labels = raw_labels - 3
    classes = len(set(map_labels))

    raw_inputs = raw_data.iloc[:, :-2].to_numpy()
    raw_inputs = (raw_inputs - np.mean(raw_inputs, axis = 0)) / np.std(raw_inputs, axis=0)
    raw_inputs = torch.tensor(raw_inputs, dtype =torch.float32)

    dataset = Wine(raw_inputs, map_labels)

    train_ratio = 0.9
    train_size = int(train_ratio*len(dataset))
    valid_size = len(dataset) - train_size

    trainset, validset = random_split(dataset, (train_size,valid_size))

    batch_size = 8
    trainloader = DataLoader(trainset, batch_size, shuffle = True)
    validloader = DataLoader(validset, batch_size, shuffle = True)

    epochs = 100
    lr = 0.01

    in_features = raw_inputs.shape[1]
    out_features = classes
    model = Model(in_features, out_features)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    for e in range(epochs):
        for batch_inputs, batch_labels in trainloader:
            loss = model(batch_inputs, batch_labels, train_mode=True)
            loss.backward()
            optim.step()
            optim.zero_grad()

        if e % 10 == 0:
            acc, valid_loss = run_valid()
            print(f"epoch:{e}, loss:{loss}, valid loss:{valid_loss}, acc:{acc*100}")
        


    pass