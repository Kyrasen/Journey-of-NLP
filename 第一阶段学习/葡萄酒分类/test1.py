import pandas as pd
import numpy as np

def onehot(labels, classes):
    res = np.zeros((len(labels), classes), dtype=int)
    rows = np.arange(len(labels))
    res[rows, labels] = 1
    return res

class Dataset:
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
    
class DataLoader:
    def __init__(self, dataset:Dataset, batch_size = 32, shuffle = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))
        
    def __iter__(self):
        self.cursor = 0
        if self.shuffle:
            np.random.shuffle(self.indices) # 直接就地打乱数组
        return self

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        batch_idx = self.indices[self.cursor: min(len(self.dataset), self.cursor+self.batch_size)]
        batch_inputs = self.dataset.inputs[batch_idx]
        batch_labels = self.dataset.labels[batch_idx]

        self.cursor += self.batch_size
        return batch_inputs, batch_labels
    
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))


def softmax(outputs):
    exp_val = np.exp(outputs)
    sum_val = np.sum(exp_val, axis = 1, keepdims = True)

    return exp_val / sum_val

def run_valid():
    current_nums = 0
    sum_loss = 0
    for batch_inputs, batch_labels in validloader:
        output = batch_inputs @ W1
        predicts = softmax(output)
        loss = -np.sum(batch_labels*np.log(predicts))/ len(batch_labels)
        sum_loss += loss
        current_nums += np.sum(np.argmax(predicts, axis= 1) == np.argmax(batch_labels, axis= 1))

    acc = current_nums / len(validloader.dataset)

    return acc, sum_loss/len(validloader)



if __name__ == "__main__":
    raw_data = pd.read_csv("data/19-WineQT.csv")
    raw_labels = raw_data["quality"].to_numpy()
    map_labels = raw_labels - 3
    classes = len(set(map_labels))
    onehot_labels = onehot(map_labels, classes)

    raw_inputs = raw_data.iloc[:, :-2].to_numpy()
    inputs = (raw_inputs - np.mean(raw_inputs, axis = 0) )/ np.std(raw_inputs, axis = 0)

    train_ratio = 0.9
    train_size = int(train_ratio * len(inputs))

    train_inputs = inputs[:train_size]
    train_labels = onehot_labels[:train_size]

    valid_inputs = inputs[train_size:]
    valid_labels = onehot_labels[train_size:]

    trainset = Dataset(train_inputs, train_labels)
    validset = Dataset(valid_inputs, valid_labels)

    batch_size = 8
    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    validloader = DataLoader(validset, 16, shuffle = True)

    in_features = train_inputs.shape[1] # 有多少个特征
    out_features = classes
    W1 = np.random.normal(size=(in_features, out_features))

    epochs = 100
    lr = 0.01

    for e in range(epochs):
        for batch_inputs, batch_labels in trainloader:
            output = batch_inputs @ W1
            predicts = softmax(output)
            loss = -np.sum(batch_labels*np.log(predicts))/ len(batch_labels)

            G = predicts - batch_labels
            dw1 = batch_inputs.T @ G
            W1 = W1 - lr*dw1

        acc, valid_loss = run_valid()
        print(f"loss:{loss}, acc:{acc*100}, valid_loss:{valid_loss}")


    pass
    