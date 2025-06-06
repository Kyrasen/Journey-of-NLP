import pandas as pd 
import numpy as np

"""
    @author: ysy
    @date: 2025/3/6
    @function: 定义 sigmoid函数 和 二元交叉熵损失函数
    @params:
        y_pred: 预测值
"""

class Dataset:
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
    
class DataLoader:
    def __init__(self, dataset: Dataset, batch_size=8, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indice = np.arange(len(self.dataset))

    def __iter__(self):
        self.cursor = 0
        if self.shuffle:
            np.random.shuffle(self.indice)
        return self
    
    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        
        actual_bs = min(self.batch_size, len(self.dataset)-self.cursor)
        actual_idx = self.indice[self.cursor: self.cursor + actual_bs]

        batch_inputs = self.dataset.inputs[actual_idx]
        batch_labels = self.dataset.labels[actual_idx]

        self.cursor += actual_bs
        return batch_inputs, batch_labels
    
    def __len__(self):
        return len(self.dataset) / self.batch_size
    
def standard_deviation(raw_inputs):
    return (raw_inputs - np.mean(raw_inputs, axis = 0)) / np.std(raw_inputs, axis = 0)

class Sigmoid:
    '''__call__让类的实例变得像函数一样可以调用，
    写法简单又符合习惯，
    非常适合封装像激活函数、损失函数这种“输入->输出”的逻辑
    '''

    def __init__(self, clip = 15):
        self.clip = clip

    def forward(self, z):
        z = np.clip(z, -self.clip, self.clip)
        return 1 / (1 + np.exp(-z))

    def __call__(self, y_pred):
        return self.forward(y_pred)

class BCELoss:
    def forward(self, predicts, labels):
        loss = -(labels*np.log(predicts) + (1-labels)*np.log(1-predicts))
        return loss
    
    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

class Model:
    def __init__(self, in_features, out_features):
        self.W = np.random.normal(size=(in_features, out_features))
        
        self.sigmoid = Sigmoid()
        self.bce_loss = BCELoss()

    def forward(self, batch_inputs, batch_labels, train_mode=True):
        z = batch_inputs @ self.W
        predicts = self.sigmoid(z)
        loss = np.mean(self.bce_loss(predicts, batch_labels))

        self.predicts = predicts
        self.batch_labels = batch_labels
        self.batch_inputs = batch_inputs

        if train_mode:
            return loss
        else:
            return predicts, loss
        
    def backward(self):
        G = (self.predicts - self.batch_labels) / len(self.batch_labels)
        DW = self.batch_inputs.T @ G
        self.W = self.W - lr * DW

    def __call__(self, batch_inputs, batch_labels, train_mode=True):
        return self.forward(batch_inputs, batch_labels, train_mode)
    

def sigmoid(y_pred):
    return 1 / (1 + np.exp(-y_pred))

def bce_loss(s, label, clip = 1e-7):
    s = np.clip(label, clip,1 - clip)
    return -(label*np.log(s) + (1-label)*np.log(1-s))

class BCEloss:
    def __call__(self,label, s, clip = 1e-7):
        s = np.clip(label, clip, 1 - clip)
        return -(label*np.log(s) + (1-label)*np.log(1-s))
    
def run_valid():
    sum_loss = 0
    correct_nums = 0
    for valid_input, valid_label in zip(valid_inputs, valid_labels):
        # valid_input = valid_input.reshape(batch_size, -1)
        # valid_label = valid_label.reshape(batch_size, -1)

        z = valid_input @ model.W
        # p = 1 / (1 + np.exp(-z))
        p = sigmoid(z)

        valid_loss = -(valid_label*np.log(p) + (1 - valid_label)*np.log(1 - p))
        sum_loss += valid_loss

        if p >= 0.5:
            res = 1
        else:
            res = 0
        if res == valid_label:
            correct_nums += 1
    acc = correct_nums/len(valid_inputs)
    return float(acc), float(sum_loss/len(validset))

if __name__ == "__main__":
    np.random.seed(110)

    raw_dataset = pd.read_csv("data/titanic/train.csv")
    raw_dataset = raw_dataset.drop(["PassengerId", "Pclass", "Name", "Sex", "Ticket", "Cabin", "Embarked"], axis = 1).dropna()
    raw_dataset = raw_dataset.sample(frac=1.0).reset_index(drop = True)

    # 取出数据
    raw_inputs = raw_dataset.iloc[:, 1:]
    raw_labels = raw_dataset.iloc[:, 0]

    # 对特征进行归一化处理
    raw_inputs = standard_deviation(raw_inputs).to_numpy()

    # 训练
    train_ratio = 0.9
    train_size = int(len(raw_inputs) * train_ratio)

    train_inputs = raw_inputs[:train_size]
    train_labels = raw_labels[:train_size].to_numpy().reshape(-1, 1)

    valid_inputs = raw_inputs[train_size:]
    valid_labels = raw_labels[train_size:].to_numpy().reshape(-1, 1)

    trainset = Dataset(train_inputs, train_labels)
    validset = Dataset(valid_inputs, valid_labels)

    epochs = 100
    lr = 0.001
    in_features = train_inputs.shape[1]
    out_features = 1
    W = np.random.normal(size=(in_features, out_features))
    batch_size = 8

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False)

    in_features = train_inputs.shape[1]
    out_features = 1

    model = Model(in_features, out_features)
    for e in range(epochs):
        for batch_inputs, batch_labels in trainloader:
            loss = model(batch_inputs, batch_labels)
            model.backward()

        acc, valid_loss = run_valid()
        print(f"Epoch: {e}, Train Loss: {loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {acc:.4f}")
    pass
