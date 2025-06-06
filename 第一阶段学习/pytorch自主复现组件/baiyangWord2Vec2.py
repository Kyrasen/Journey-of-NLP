import re
import jieba
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch
import pickle

def read_data(file):        
    with open(file, "r", encoding="utf-8") as f:
        data = f.read()
        raw_data = re.sub("\u3000|\n", "", data)

    return raw_data

def load_stopwords(file):
    with open(file, "r", encoding="utf-8") as f:
        stopwords = f.read().split("\n")

    return stopwords

def tokenize(raw_corpus, stopwords):
    vocab = {"[PAD]":0, "[UNK]":1}
    raw_tokens = jieba.lcut(raw_corpus)
    tokens = []
    for token in raw_tokens:
        if token not in stopwords:
            vocab.setdefault(token, len(vocab))
            tokens.append(token)

    return vocab, tokens

class BaiYang(Dataset):
    def __init__(self, positive_pairs, negative_pairs):
        self.dataset = positive_pairs + negative_pairs

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        center, context, label = self.dataset[idx]
        return (torch.tensor(vocab[center], dtype=torch.long),
                torch.tensor(vocab[context], dtype=torch.long),
                torch.tensor(label, dtype=torch.float))
    

def building_dataset(tokens,vocab ,win_size = 6, neg_num = 6):
    
    positive_samples = []
    negative_samples = []
    ids = list(vocab.keys())
    # 正样本
    for idx,token in enumerate(tokens):
        left = max(idx - win_size, 0)
        right = min(len(tokens), idx+win_size)
        contexts = ids[left+2:right+2]
        for context in contexts:
            positive_samples.append([token, context, 1])
        
        # 负样本
        neg_samples = []
        while len(neg_samples) < neg_num:
            negSample = np.random.choice(ids)
            if negSample in contexts:
                continue
            elif (negSample == "[PAD]" or negSample == "[UNK]"):
                continue
            else:
                neg_samples.append(str(negSample))
        for neg_sample in neg_samples:
            negative_samples.append([token, neg_sample, 0])
    return positive_samples, negative_samples

class Model(nn.Module):
    def __init__(self, in_features, emb_dim):
        super().__init__()
        self.center_emb = nn.Embedding(in_features, emb_dim)
        self.context_emb = nn.Embedding(in_features, emb_dim)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch_center, batch_context, batch_labels):
        center_emb = self.center_emb(batch_center)
        context_emb = self.context_emb(batch_context)
        scores = torch.sum(center_emb * context_emb, dim=1)

        loss = self.loss_fn(scores, batch_labels)
        

        return loss



if __name__ == "__main__":

    baiyang_file = "data/白杨礼赞.txt"
    stopwords_file = "data/mystopWords.txt"

    raw_data = read_data(baiyang_file)
    stopwords = load_stopwords(stopwords_file)

    vocab, tokens = tokenize(raw_data,stopwords )      # 返回词典，分词集

    # 获取正负样本    
    positive_pairs, negative_pairs = building_dataset(tokens,vocab)

    batch_size = 16
    trainset = BaiYang(positive_pairs,negative_pairs)
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)

    lr = 0.01
    epochs = 50
    in_features = len(vocab)
    emb_dim = 64
    model = Model(in_features, emb_dim)
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    for e in range(epochs):
        for batch_center, batch_context,    batch_labels in trainloader:
            loss = model(batch_center, batch_context, batch_labels)
            optim.step()
            optim.zero_grad()

        print(f"epoch:{e}, loss:{loss}")

    with open("data/Baiyangvocab.pt", "wb") as f:
        pickle.dump((model.center_emb.weight.data.numpy(), vocab),f)
        print("save embedding and vocab")

    pass