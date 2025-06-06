'''
    古诗生成：自己写的，RNN：循环网络
'''
import pandas as pd
import re
import json

from sympy.integrals.risch import NonElementaryIntegral
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn

def read_data(file):
    df = pd.read_csv(file)
    content = df["内容"].dropna()
    poems = []
    for poem in content:
        words = re.findall(r"[\u4e00-\u9fff]", poem)    # 提取每个诗中的中文字符，这里返回的是一个列表
        poems.append(words)
    return poems

def tokenizer(poems):
    vocab = {}
    idx2word = {}
    for poem in poems:
        for word in poem:
            vocab.setdefault(word, len(vocab))
    idx2word = {v:k for k, v in vocab.items()}
    return vocab, idx2word

def build_dataset(vocab, poems):
    texts = []
    labels = []
    for poem in poems:
        for idx,word in enumerate(poem):
            text = poem[:idx+1]
            try:
                label = poem[idx+1]
            except:
                continue
            texts.append(text)
            labels.append(label)

    return texts, labels

class MyDataset(Dataset):
    def __init__(self, texts, labels, num = None):
        self.texts = texts
        self.labels = labels
        self.num = num
    def __len__(self):
        if self.num:
            return self.num
        else:
            return len(self.texts)
    def __getitem__(self, idx):  # 这里返回的是tokens_id, label
        text = self.texts[idx]
        label = self.labels[idx]
        tokens_id = [vocab[word] for word in text]
        if len(tokens_id) >= seq_len:
            tokens_id = tokens_id[:seq_len]
        else:
            tokens_id += (seq_len - len(tokens_id)) * [0]
        label_id = vocab[label]
        return torch.tensor(tokens_id, dtype=torch.long), label_id

class RNN(torch.nn.Module):
    def __init__(self, hidding_num):
        super().__init__()
        self.W = torch.nn.Linear(hidding_num, hidding_num)
        self.U = torch.nn.Linear(hidding_num, hidding_num)
        self.V = torch.nn.Linear(hidding_num, hidding_num)
        self.relu = torch.nn.ReLU()

    def forward(self, emb):     # emb : batch_size * seq_len * emb_dim
        results = torch.zeros(emb.shape)    # results : batch_size * seq_len * emb_dim
        p = torch.zeros(size = (emb.shape[0], emb.shape[2]))      # s0 : seq_len * emb_dim
        # s = torch.zeros(size=(emb.shape[0], emb.shape[2]))
        for i in range(seq_len):    # 每个字遍历
            x_v = self.V(emb[:, i , :])
            p_w = self.W(p)
            s = x_v + p_w
            # s[i] = s
            s = self.relu(s)
            out_put = self.U(s)
            p = s
            results[:, i, :] = out_put
        return results

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.rnn = RNN(emb_dim)
        self.classify = torch.nn.Linear(emb_dim, vocab_size)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, batch_token_ids, batch_labels = None):
        # batch_token_ids : batch_size * seq_len
        emb = self.embedding(batch_token_ids)  # emb : batch_size * seq_len * emb_dim
        rnn_out = self.rnn(emb)  # rnn_out: batch_size * seq_len * emb_dim
        rnn_out_mean = torch.mean(rnn_out, dim=1)  # batch_size * emb_dim
        predicts = self.classify(rnn_out_mean)  # batch_size * label_nums
        if batch_labels is not None:
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            loss = self.loss_fn(predicts, batch_labels)
            return loss
        return predicts


def generate_poem(start_word, model):
    model.eval()
    start_word_id = torch.tensor([[vocab[start_word]]], dtype=torch.long)
    sentence = [start_word]
    with torch.no_grad():
        for _ in range(4):
            start_word_id_padding = start_word_id[0].tolist() + [0] * (seq_len - start_word_id.size(1))
            start_word_ids_tensor = torch.tensor([start_word_id_padding], dtype=torch.long)
            predicts = model(start_word_ids_tensor)
            next_id = torch.argmax(predicts, dim = 1).item()
            next_word = idx2word.get(next_id, "？")
            sentence.append(next_word)
            start_word_id = torch.cat([start_word_id, torch.tensor([[next_id]])], dim = 1)
    return "".join(sentence)

if __name__ == "__main__":
    file = "D:/Desktop/deepLearnProject/day 19/data/5poems3.csv"
    seq_len = 32
    poems = read_data(file)
    emb_dim = 756

    vocab, idx2word = tokenizer(poems)
    # with open("D:/Desktop/deepLearnProject/day 19/data/vocab.json", "w", encoding="utf-8") as f:
    #     json.dump(vocab, f)

    # with open("D:/Desktop/deepLearnProject/day 19/data/vocab.json", "r") as f:
    #     vocab = json.load(f)
    vocab_size = len(vocab)
    texts, labels = build_dataset(vocab, poems)

    trainset = MyDataset(texts, labels)
    batch_size = 32
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True)

    lr = 0.001
    model = Model()
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    epochs = 10
    for e in range(epochs):
        model.train()
        for batch_tokens_id, batch_labels in tqdm(trainloader):
            loss = model(batch_tokens_id, batch_labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
        print(f"loss : {loss}")

    heads = "布丁牛逼"
    for head in heads:
        line = generate_poem(head, model)
        print(line)
    pass