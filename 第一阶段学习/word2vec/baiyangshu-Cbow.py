import re
import jieba
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import pickle

def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data_str = f.read()
        data_str = re.sub("\n|\u3000", "", data_str)
    return data_str

def load_stopwords(file):

    with open(file, "r", encoding="utf-8") as f:
        stopwords_list = f.read().split("\n")

    return set(stopwords_list)

def tokenize(data_str, stopwords_list):
    tokens_list = jieba.lcut(data_str)
    vocab = {}
    for token_str in tokens_list:
        if token_str not in stopwords_list:
            vocab.setdefault(token_str, len(vocab))

    return vocab, list(vocab.values())

def build_negSamples(contextId_list, negNum_int, vocabLen_int):
    negSamplesId_list = []
    while( len(negSamplesId_list) < negNum_int):
        randomNum = random.randint(0, vocabLen_int-1)
        if randomNum not in contextId_list:
            negSamplesId_list.append(randomNum)
        else:
            continue
    return negSamplesId_list

def build_dataset(vocab_dict, tokenIds_list, winSize_int = 10, negNum_int = 5):
    positiveSamples_list2   = []
    negSamples_list2        = []
    wholeSamples_list2      = []
    for centerId_int in tqdm(tokenIds_list, desc = "building dataset"):
        contextId_list = tokenIds_list[max(centerId_int-winSize_int, 0) : min(len(tokenIds_list), centerId_int+winSize_int)]

        # 正样本
        positiveSamples_list2.extend(zip(contextId_list, [centerId_int]*winSize_int, [1]*winSize_int))
        wholeSamples_list2.extend(positiveSamples_list2)
        # 负样本
        negSamples_list = build_negSamples(contextId_list, negNum_int, len(vocab_dict))
        negSamples_list2.extend(zip(negSamples_list, [centerId_int]*winSize_int, [0]*winSize_int))
        
        wholeSamples_list2.extend(negSamples_list2)

    return wholeSamples_list2

class Baiyang(Dataset):
    def __init__(self, tripleSamplesId_list):
        self.dataset = tripleSamplesId_list

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        contextID_int, centerId_int, label_int = self.dataset[idx]
        return contextID_int, centerId_int, label_int
    
class CBoW(nn.Module):
    def __init__(self, vocabLen_int, embDim_int):
        super().__init__()
        self.center_emb = nn.Embedding(vocabLen_int, embDim_int)
        self.context_emb = nn.Embedding(vocabLen_int, embDim_int)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batchContextId_int, batchCenterId_int, batchLabel_int):
        centerEmb_tensor = self.center_emb(batchCenterId_int)
        contextEmb_tensor = self.context_emb(batchContextId_int)

        similarities_tensor = torch.sum(centerEmb_tensor*contextEmb_tensor, dim=1)
        loss = self.loss_fn(similarities_tensor, batchLabel_int)

        return loss

if __name__ == "__main__":
    baiyang_file = "day12/data/白杨礼赞.txt"
    stopwords_file = "day10/data/mystopWords.txt"

    data_str = read_data(baiyang_file)
    stopwords_list = load_stopwords(stopwords_file)

    vocab_dict, tokenIds_list = tokenize(data_str, stopwords_list)
    
    tripleSamplesId_list = build_dataset(vocab_dict, tokenIds_list)
    
    trainset = Baiyang(tripleSamplesId_list)
    batch_size_int = 8
    trainloader = DataLoader(trainset, batch_size = batch_size_int, shuffle=True)
    embDim_int = 64
    model = CBoW(len(vocab_dict), embDim_int)

    epochs = 10
    lr = 0.01
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    for e in range(epochs):
        for batchContextId_int, batchCenterId_int, batchLabel_int in tqdm(trainloader, desc = "moding training"):
            batchContextId_int = batchContextId_int.to(torch.long)
            batchCenterId_int = batchCenterId_int.to(torch.long)
            batchLabel_int = batchLabel_int.to(torch.float)
            loss = model(batchContextId_int,batchCenterId_int , batchLabel_int)

            loss.backward()
            optim.step()
            optim.zero_grad()

        print(f"epoch:{e}, loss:{loss}")

    with open("day12/data/baiyangshuCBOW.pt", "wb") as f:
        pickle.dump((model.center_emb.weight.data.numpy(), vocab_dict), f)

    pass