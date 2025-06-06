import re

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.read().split("\n")

    pattern = re.compile(r"^(.*?)\t(.*?)\t")
    processed_data = []
    for line in data:
        matches = pattern.match(line)
        if matches:
            english, chinese  = matches.groups()
            processed_data.append([chinese, english])
    return processed_data

class MyDataset(Dataset):
    def __init__(self, data, num = None):
        self.data = data    # 二维列表，[[zh1, en1],[zh2, en2], ...]
        self.num = num

    def __len__(self):
        if self.num:
            return self.num
        return len(self.data)

    def __getitem__(self, idx):
        chinese = self.data[idx][0]
        english = self.data[idx][1]

        chinese_ids = [chinese_word_ids.get(word, chinese_word_ids["U"]) for word in chinese]
        english = english.split(" ")
        english_ids = [english_word_ids.get(word, english_word_ids["U"]) for word in english]

        texts_id = chinese_ids + english_ids
        labels_ids = english_ids
        chinese_nums = len(chinese_ids)
        if len(texts_id) < max_len:
            texts_id += [chinese_word_ids["P"]] * (max_len - len(texts_id))
        else:
            texts_id = texts_id[:max_len]
        return  torch.tensor(texts_id, dtype=torch.long), torch.tensor(labels_ids, dtype=torch.long)

def tokenizer(data):
    chinese_word_ids = {"U":0, "P":1}     # 以字分词
    english_word_ids = {"E":3, "U":2}     # 以单词分词
    for line in data:
        for chinese in line[0]:
            for chineseWord in chinese:
                chinese_word_ids.setdefault(chineseWord, len(chinese_word_ids))
        words = line[1].split(" ")
        for english in words:
            for englishWord in words:
                english_word_ids.setdefault(englishWord, len(chinese_word_ids))

    chinese_ids_word = {k: v for v, k in chinese_word_ids.items()}
    english_ids_word = {k: v for v, k in english_word_ids.items()}

    return chinese_word_ids, chinese_ids_word, english_word_ids, english_ids_word

class Model(torch.nn.Module):
    def __init__(self, emb_dim, vocab_size):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, emb_dim)
        self.rnn = torch.nn.RNN(emb_dim, emb_dim, batch_first=True)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, batch_texts_ids, batch_labels_ids):
        emb = self.embeddings(batch_texts_ids)
        rnn_out = self.rnn(emb)
        cls = rnn_out @ self.embeddings.weight.data.T

        if batch_labels_ids is not None:
            loss = self.loss_fn(cls, batch_labels_ids)
            return loss
        else:
            return cls


if __name__ == "__main__":
    file = "../data/cmn.txt"
    data = read_data(file)  # 二维列表，[[zh1, en1],[zh2, en2], ...]
    max_len = 10
    chinese_word_ids, chinese_ids_word, english_word_ids, english_ids_word = tokenizer(data)
    vocab_size = len(chinese_word_ids) + len(english_ids_word)
    epochs = 10
    lr = 1e-3
    emb_dim = 256
    batch_size = 10

    trainset = MyDataset(data, 100)
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True)

    model = Model(emb_dim, vocab_size)
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    for e in range(epochs):
        for batch_texts_ids, batch_labels_ids in tqdm(trainloader):
            loss = Model(batch_texts_ids, batch_labels_ids)
            loss.backward()
            optim.step()
            optim.zero_grad()
        print(f"loss : {loss}")
