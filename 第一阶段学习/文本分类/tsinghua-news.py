from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import re
import numpy as np
from scipy.sparse import identity
import torch.nn as nn

def evaluate(model,validloader):
    model.eval()
    total_loss = 0
    correct = 0
    total_num = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in validloader:
            predicts, loss = model(batch_inputs, batch_labels, train_mode=False)

            predictions = torch.argmax(predicts, dim=1)
            labels = torch.argmax(batch_labels, dim=1)

            correct += (predictions == labels).sum().item()
            total_num += batch_labels.shape[0]

            total_loss += loss.item()

    avg_loss = total_loss / len(validloader)
    accuracy = correct / total_num
    print(f"Validation Loss: {avg_loss}, accuracy: {accuracy}")
    model.train()


def collate_fn(batch):
    sentences, labels = zip(*batch) # 拆分batch
    sentences = [torch.tensor(s, dtype=torch.float32) for s in sentences]
    sentences_padded = pad_sequence(sentences, batch_first = True, padding_value=0)

    labels = torch.tensor(labels, dtype = torch.float32) # 转换labels

    return sentences_padded, labels

def onehot(labels, classes = 4):
    res = np.zeros((len(labels), classes))
    rows = np.arange(len(labels))
    res[rows,labels] = 1

    return res

def tokenize(corpus):
    vocab = {"PAD":0, "UNK":1}
    for line in corpus:
        for word in line:
            vocab.setdefault(word, len(vocab))
    return vocab

def encode(sentence, max_len = 20):
    sentence = sentence[:max_len]
    wordIdx = [vocab.get(word, vocab["UNK"]) for word in sentence]
    wordIdx += [vocab["PAD"]] * (max_len - len(sentence))
    sentence_emb = whole_emb[wordIdx]

    return sentence_emb

class NewsDataset(Dataset):
    def __init__(self, corpus, labels):
        super().__init__()

        self.corpus = corpus
        self.labels = labels
        self.vocab = {"PAD":0, "UNK":1}
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        sentence = self.corpus[idx]
        return encode(sentence), self.labels[idx]
    
    def encode(self,sentence, max_len=20):
        sentence = sentence[:max_len]
        wordIdx = [vocab[word] for word in sentence]
        wordIdx += [vocab["PAD"]] * (max_len - len(sentence))


def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        print(f"fist few lines:{lines[:5]}")
        pattern = re.compile(r"(.+)\s(\d+)")
        corpus = []
        labels = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            matches = pattern.findall(line)
            for title, label in matches:
                corpus.append(title)
                labels.append(int(label))
        print(f"Loaded {len(corpus)} sentences and {len(labels)} labels.") 

    return corpus, labels

def process(labels):
    label_map = {3:0, 6:1, 7:2, 8:3}
    return [label_map.get(l,l) for l in labels]

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, labels, train_mode=True):
        output = self.linear1(inputs)
        sentence_vec = torch.mean(output, dim=1)
        loss = self.loss_fn(sentence_vec, labels)

        if train_mode:
            return loss
        else:
            predicts = torch.softmax(sentence_vec, dim=1)
            return predicts, loss
        
if __name__ == "__main__":
    train_file = "data/train2.txt"
    valid_file = "data/test2.txt"

    train_corpus, train_labels = read_data(train_file)
    train_labels = process(train_labels)
    valid_corpus, valid_labels = read_data(valid_file)
    valid_labels = process(valid_labels)

    vocab = tokenize(train_corpus)

    classes = len(set(train_labels))
    train_labels = onehot(train_labels)
    valid_labels = onehot(valid_labels)
    
    wordNum = len(vocab)
    whole_emb = identity(wordNum, dtype=np.float32).toarray()

    trainset = NewsDataset(train_corpus, train_labels)
    validset = NewsDataset(valid_corpus, valid_labels)

    batch_size = 8
    trainloader = DataLoader(trainset, batch_size, shuffle = True, collate_fn = collate_fn)
    validloader = DataLoader(validset, batch_size, shuffle = False, collate_fn = collate_fn)

    in_features = wordNum
    out_features = classes

    epochs = 100
    lr = 0.05
    model = Model(in_features, out_features)
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    for e in range(epochs):
        for batch_inputs, batch_labels in tqdm(trainloader):
            loss = model(batch_inputs, batch_labels)
            loss.backward()

            optim.step()
            optim.zero_grad()
        print(f"epoch:{e} ,loss : {loss}")

        if e % 10 == 0:
            evaluate(model, validloader)

    pass
