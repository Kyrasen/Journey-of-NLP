import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import random

def is_five_character_quatrain(poem):
    lines = re.split(r"[,，.。!！?？:：;；\n]", poem)
    lines = [line.strip() for line in lines if line.strip()]
    return len(lines) == 4 and all( len(line) == 5 for line in lines)

def read_data(file):

    df = pd.read_csv(file, encoding="utf-8")
    poems = df["内容"].dropna()
    df_five = df[df["内容"].apply(is_five_character_quatrain)]
    
    return df_five

def tokenizer(all_poems):
    word_2_idx = {"PAD":0, "UNK":1, "EOS":2, "BOS":3}

    for poem in all_poems:
        for word in poem:
            word_2_idx.setdefault(word, len(word_2_idx))

    return word_2_idx

class MyDataset(Dataset):
    def __init__(self, all_poems):
        self.all_poems = all_poems

    def __len__(self):
        return len(self.all_poems)
    
    def __getitem__(self, idx):
        poem = self.all_poems[idx]
        poem_ids = [word_2_idx.get(word, word_2_idx["UNK"]) for word in poem]

        input_ids = poem_ids.copy()     # 不需要补齐或者截断，因为每个Input都是一样的长度，然后如果需要处理文本长度的话，只需要input的句子长度一样或者label的长度是一样的，就行了，不需要input 和 label 的长度是一样的

        label_ids = input_ids[1:] + [word_2_idx["EOS"]]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)
    
class RNN(torch.nn.Module):
    def __init__(self, in_features, out_features, batch_first = True):
        super().__init__()
        self.W = torch.nn.Linear(out_features, out_features)    # h_{t_1} -> hidden
        self.U = torch.nn.Linear(in_features, out_features)   # x_t -> hidden 
        self.out_features = out_features
        
    def forward(self, input_emb):
        batch_size, seq_len, _ = input_emb.shape
        h_0 = torch.zeros(size = (batch_size, self.out_features))
        h_p = h_0
        outputs = torch.zeros(size = (batch_size, seq_len, self.out_features))
        for i in range(seq_len):
            U_xt = self.U(input_emb[:, i, :])
            W_h_p = self.W(h_p)
            h = torch.tanh((U_xt + W_h_p))
            outputs[:, i, :] = h
            h_p = h
        return outputs, h_p

class Model(torch.nn.Module):
    def __init__(self,input_emb_dim, hidden_size, vocab_size):
        super().__init__()
        self.input_emb = torch.nn.Embedding(vocab_size, input_emb_dim)
        self.rnn = RNN(input_emb_dim, hidden_size, batch_first = True)
        
    
    def forward(self, batch_input_ids):
        input_emb = self.input_emb(batch_input_ids) # (batch_size, seq_len, input_emb_dim)
        rnn_out, last_hidden = self.rnn(input_emb)  # rnn_out(batch_size, seq_len, hidden_size)
        cls = rnn_out @ self.input_emb.weight.T # 如果要参数共享的话，hidden_size = input_emb_dim, cls.shape: (batch_size, seq_len, vocab_size)
        return cls
    
def decode(ids, idx_2_word):
    result = []
    for id in ids:
        result.append(idx_2_word[id])
    return "".join(result)

def random_gen_poem():
    model.eval()
    ids = [random.choice(list(word_2_idx.values()))]
    with torch.no_grad():
        for i in range(seq_len):
            cls = model(torch.tensor([ids], dtype=torch.long))  # 这里要传进2D或3D的维度
            pro_id = torch.argmax(cls[0, -1, :], dim=-1).item()
            ids.append(pro_id)
    return ids

if __name__ == "__main__":
    file = "day19/data/元.csv"

    all_poetries = read_data(file)["内容"].tolist()

    word_2_idx = tokenizer(all_poetries)
    idx_2_word = {k:v for v, k in word_2_idx.items()}
    vocab_size = len(word_2_idx)
    seq_len = 24

    # 调参区
    batch_size = 32
    epochs = 100
    lr = 5e-3
    input_emb_dim = 100
    hidden_size = input_emb_dim


    trainset = MyDataset(all_poetries)
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True)

    model = Model(input_emb_dim, hidden_size, vocab_size)
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = torch.nn.CrossEntropyLoss()   # 传的参数一定要是二维的

    for e in range(epochs):
        for batch_input_ids, batch_label_ids in tqdm(trainloader, desc = "Training ..."):
            cls = model(batch_input_ids)
            # batch_input_ids.shape:(batch_size, seq_len)
            # cls.shape: (batch_size, seq_len, vocab_size)
            loss = loss_fn(cls.view(-1, vocab_size), batch_label_ids.view(-1))

            loss.backward()
            optim.step()
            optim.zero_grad()
        if e % 10 == 0:
            print(f"e : {e}, loss : {loss}")

    while True:
        ids = random_gen_poem()
        result = decode(ids,idx_2_word)
        print(result)

    pass