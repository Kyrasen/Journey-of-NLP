import re
import jieba
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import pickle

def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.read()

    data = re.sub("\u3000","",data) # re.sub()将字符串中的"\u3000"用""来替换
    data = data.split("\n")
    return data

def loadStopword(file):
    with open(file, "r", encoding="utf-8") as f:
        stopwords = f.read().split("\n")

    return stopwords

def tokenize(raw_data):
    vocab = {"[PAD]":0, "[UNK]":1}
    token_corpus = []
    for paragraph in raw_data:
        tokens = jieba.lcut(paragraph)
        tokens = [token for token in tokens if token not in stopwords]
        token_corpus.append(tokens)
        for token in tokens:
            vocab.setdefault(token, len(vocab))

    return vocab, token_corpus

def build_dataset(token_corpus, win_size = 6):
    pairs = []
    for token_sentence in token_corpus:
        ids = [vocab.get(token, vocab["[UNK]"]) for token in token_sentence]    # 先把每个token的索引取出来
        for idx, center in enumerate(ids):      # 对每句话进行处理，重点是形成配对
            contexts = ids[max(idx-win_size, 0) : idx] + ids[idx+1: min(idx+win_size, len(token_sentence))]
            pairs.extend([(center, context) for context in contexts])
        
    return pairs

class Baiyang(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return pair[0], pair[1]
    
class Model(nn.Module):
    def __init__(self, in_features, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(in_features, emb_dim)
        self.W = nn.Linear(emb_dim, in_features)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_inputs, batch_labels):
        # batch_inputs = batch_inputs.float()
        z = self.emb(batch_inputs)
        output = self.W(z)
        loss = self.loss_fn(output, batch_labels)

        return loss


if __name__ == "__main__":
    baiyang_file = "data/白杨礼赞.txt"
    stopwords_file = "data/mystopWords.txt"

    stopwords = loadStopword(stopwords_file)
    raw_data = read_data(baiyang_file)      # raw_data是list

    vocab, token_corpus = tokenize(raw_data)       

    pairs = build_dataset(token_corpus)       

    trainset = Baiyang(pairs)
    batch_size = 16

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    epochs = 10
    lr = 0.01
    in_features = len(vocab)
    emb_dim = 64

    model = Model(in_features, emb_dim)
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    for e in range(epochs):
        for batch_inputs, batch_labels in tqdm(trainloader):
            loss = model(batch_inputs, batch_labels)

            loss.backward()
            optim.step()
            optim.zero_grad()

        print(f"epoch:{e} ,loss:{loss}")

    
    with open("data/baiyang.pt", "wb") as f:
        pickle.dump((vocab, model.emb.weight.data.numpy()), f)
        print("vocab and emb_token have saved.")

    pass
