
import re
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm

def read_data(file):
    corpus = []
    labels = []
    with open(file, "r", encoding = "utf-8") as f:
        lines = f.read().strip().split("\n")
        pattern = re.compile(r"(.+)\s(\d+)")
        for line in lines:
            matches = pattern.findall(line)
            for title, label in matches:
                corpus.append(title)
                labels.append(int(label))
                pass

    return corpus, labels

def mapping(label):
    label_vocab = {3:0, 6:1, 7:2, 8:3}
    label = int(label)
    mapped_label = label_vocab.get(label, -1)
    if mapped_label == -1:
        print(f"[EORROR] Invalid label found: {label}")
    return mapped_label

def tokenize(corpus):
    vocab = {"[PAD]":0, "[UNK]":1}
    for sentence in corpus:
        for word in sentence:
            vocab.setdefault(word, len(vocab))

    return vocab

def encode(sentence, seq_len=20):
    sentence = sentence[:seq_len]
    wordIdx = [vocab.get(word, vocab["[UNK]"]) for word in sentence]

    # print(f"Max word index:{max(wordIdx)}, Vocab size: {len(vocab)}")
    return wordIdx


class Tsing(Dataset):
    def __init__(self, corpus, labels, seq_len=20):
        self.corpus = corpus
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        label_emb = mapping(self.labels[idx])
        if label_emb == -1:
            raise ValueError(f"Invalid label found:{self.labels[idx]}")
        
        wordIdx = encode(self.corpus[idx])
        wordIdx += [vocab["[PAD]"]] * (self.seq_len - len(wordIdx))
        # wordIdx = np.array(wordIdx)
        # sentence_emb = random_emb[wordIdx]

        return torch.tensor(wordIdx, dtype=torch.int), torch.tensor(label_emb, dtype=torch.long)
    
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), emb_dim)
        self.W1 = nn.Linear(in_features, 256)
        self.W2 = nn.Linear(256, out_features)
        self.relu = nn.ReLU()

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, labels, train_mode=True):
        sentence_emb = self.emb(inputs)
        z1 = self.W1(sentence_emb)
        z1 = self.relu(z1) # 防止梯度爆炸
        z2 = self.W2(z1)
        z2 = torch.mean(z2, dim=1)

        loss = self.loss_fn(z2, labels)

        if train_mode:
            return loss
        else:
            predicts = torch.softmax(z2, dim=1)
            return predicts, loss
        
def evaluate():
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in validloader:
            predicts, loss = model(batch_inputs, batch_labels, train_mode=False)

            correct += torch.sum(torch.argmax(predicts, dim=1) == batch_labels)
            total_loss += loss.detach()

        acc = correct / len(validloader.dataset)
        loss = total_loss / len(validloader)

    return acc, loss

if __name__ == "__main__":
    train_file = "data/train2.txt"
    valid_file = "data/test2.txt"

    train_corpus, train_labels = read_data(train_file)
    valid_corpus, valid_labels = read_data(valid_file)
    classes = len(set(train_labels))
    vocab = tokenize(train_corpus)

    trainset = Tsing(train_corpus, train_labels)
    validset = Tsing(valid_corpus, valid_labels)
    emb_dim = 64

    random_emb = np.random.normal(size=(len(vocab), emb_dim))  # 每个字由64个数编码组成, 这里的编码数就是前面我们所学的特征数

    batch_size = 8
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size = 4, shuffle=False)

    epochs = 50
    lr = 0.01

    model = Model(in_features=emb_dim, out_features=classes)
    optim = torch.optim.Adam(model.parameters(),lr = lr)

    for e in range(epochs):
        for batch_inputs, batch_labels in tqdm(trainloader):
            loss = model(batch_inputs,batch_labels, train_mode=True)
            
            loss.backward()
            optim.step()
            optim.zero_grad()

        acc, valid_loss = evaluate()
        print(f"loss:{loss}, acc:{acc*100:.2f}%, valid_loss:{valid_loss}")

    pass