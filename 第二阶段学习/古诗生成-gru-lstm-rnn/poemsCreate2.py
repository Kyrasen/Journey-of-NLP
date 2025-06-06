'''
    古诗生成：自己写的，LSTM：循环网络
'''
import pandas as pd
import re
import time
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
    vocab = {"<pad>":0, "<eos>":1, "<unk>":2}  # 添加填充符号和结束符号
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
        poem += ["<eos>"]   # 在结尾出加上结束符号
        for idx,word in enumerate(poem[:-1]):
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


class LSTM(torch.nn.Module):
    def __init__(self, hidding_num):
        super().__init__()
        self.W = torch.nn.Linear(hidding_num, hidding_num)
        self.U = torch.nn.Linear(hidding_num, hidding_num)
        self.V = torch.nn.Linear(hidding_num, hidding_num)
        # self.relu = torch.nn.ReLU()

    def forward(self, emb):  # emb : batch_size * seq_len * emb_dim
        results = torch.zeros(emb.shape)  # results : batch_size * seq_len * emb_dim
        p1 = torch.zeros(size=(emb.shape[0], emb.shape[2]))  # p1 : batch_size * emb_dim
        p2 = torch.zeros(size=(emb.shape[0], emb.shape[2]))  # p2 : batch_size * emb_dim
        for i in range(seq_len):  # 每个字遍历
            xi = emb[:, i, :]
            xi = xi + p2

            Wxi = self.W(xi)
            Wxi_sigmoid = torch.sigmoid(Wxi)
            p1 = Wxi_sigmoid * p1

            Uxi = self.U(xi)
            Uxi_sigmoid = torch.sigmoid(Uxi)

            Vxi = self.V(xi)
            Vxi_tanh = torch.tanh(Vxi)

            UVxi = Uxi_sigmoid * Vxi_tanh

            p1 = p1 + UVxi

            p2 = torch.sigmoid(xi) * torch.tanh(p1)
            results[:, i, :] = p2

        return results

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.lstm = torch.nn.LSTM(emb_dim, emb_dim)
        self.classify = torch.nn.Linear(emb_dim, vocab_size, bias = False)

        self.classify.weight = self.embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, batch_token_ids, batch_labels = None):
        # batch_token_ids : batch_size * seq_len
        emb = self.embedding(batch_token_ids)  # emb : batch_size * seq_len * emb_dim
        rnn_out = self.lstm(emb)  # rnn_out: batch_size * seq_len * emb_dim
        rnn_out_mean = torch.mean(rnn_out, dim=1)  # batch_size * emb_dim
        predicts = self.classify(rnn_out_mean)  # batch_size * label_nums
        if batch_labels is not None:
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            loss = self.loss_fn(predicts, batch_labels)
            return loss
        return predicts


def generate_poem(start_word, model, max_len = 64):
    model.eval()
    start_word_id = torch.tensor([[vocab[start_word]]], dtype=torch.long)   # 这里的start_word_id是二维的，
    sentence = [start_word]
    with torch.no_grad():
        for _ in range(max_len):
            start_word_id_padding = start_word_id[0].tolist() + [0] * (seq_len - start_word_id.size(1))
            start_word_ids_tensor = torch.tensor([start_word_id_padding], dtype=torch.long)
            predicts = model(start_word_ids_tensor)
            next_id = torch.argmax(predicts, dim = 1).item()    # 想打印，想看值，转换成标量用item, 如果是计算或者反传，用tensor
            next_word = idx2word.get(next_id, "？")
            if next_word == "<eos>":
                break
            sentence.append(next_word)
            start_word_id = torch.cat([start_word_id, torch.tensor([[next_id]])], dim = 1)  # 在原来生成的基础上，添加新的词，这样的话才能更新输入
    return "".join(sentence)

def print_with_effect(text, delay = 0.1):
    for char in text:
        print(char, end="", flush = True)   # 这里的flush有什么用呢？确保每个字符立刻出现在终端或其他的显示器上，而不是在缓存中等待
        time.sleep(delay)
    print()

if __name__ == "__main__":
    file = "D:/Desktop/deepLearnProject/day 19/data/5poems3.csv"
    seq_len = 32
    poems = read_data(file)
    emb_dim = 1024

    vocab, idx2word = tokenizer(poems)
    # with open("D:/Desktop/deepLearnProject/day 19/data/vocab.json", "w", encoding="utf-8") as f:
    #     json.dump(vocab, f)

    # with open("D:/Desktop/deepLearnProject/day 19/data/vocab.json", "r") as f:
    #     vocab = json.load(f)
    vocab_size = len(vocab)
    texts, labels = build_dataset(vocab, poems)

    trainset = MyDataset(texts, labels,640)
    batch_size = 128
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True)

    lr = 0.005
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

    with open("D:/Desktop/deepLearnProject/day 19/data/5poems2.csv", "w", encoding="utf-8") as f:
        heads = "布丁牛逼"
        for head in heads:
            line = generate_poem(head, model)
            print_with_effect(line)
            f.write("\n" + line)
        f.write("\n")
    pass