import pandas as pd
import torch
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        df = pd.read_csv(file)["内容"]

    return list(df)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // self.num_heads

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)

        self.o_proj = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, atten_mask = None):
        batch_size, seq_len, d_model = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        K = K.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        V = V.reshape(batch_size, self.num_heads, seq_len, self.d_k)

        atten_term = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if atten_mask is not None:
            atten_term.masked_fill_(atten_mask == 0, -1e9)

        atten_scores = torch.softmax(atten_term, dim=-1) @ V
        output = self.o_proj(atten_scores.contiguous().view(batch_size, seq_len, d_model))
        
        return output
    
class FeedForward(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dff = d_model * 4
        self.W1 = torch.nn.Linear(d_model, self.dff)
        self.activition = torch.nn.ReLU()
        self.W2 = torch.nn.Linear(self.dff, d_model)

    def forward(self,x):
        x = self.W1(x)
        x = self.activition(x)
        x = self.W2(x)

        return x

class LayerNorm(torch.nn.Module):
    def __init__(self, d_model, esp = 1e-3):
        super().__init__()

        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.beta = torch.nn.Parameter(torch.zeros(d_model))

        self.esp = esp

    def forward(self,x):
        mean = torch.mean(x, dim=-1, keepdim = True)
        std = torch.std(x, dim=-1, keepdim = True)

        output = ((x - mean) / (std + self.esp)) * self.gamma + self.beta

        return output


class DecoderBlock(torch.nn.Module):
    def __init__(self,d_model, num_heads, dropout=0.1):
        super().__init__()
        self.mha_block1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.mha_block2 = MultiHeadAttention(d_model, num_heads, dropout)
        self.layerNorm1 = torch.nn.LayerNorm(d_model)
        self.layerNorm2 = torch.nn.LayerNorm(d_model)
        self.layerNorm3 = torch.nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x, atten_mask=None):
        norm1_out = self.layerNorm1(x + self.mha_block1(x, atten_mask))
        norm2_out = self.layerNorm2(norm1_out + self.mha_block2(norm1_out, atten_mask))
        norm3_out = self.layerNorm3(norm2_out + self.ffn(norm2_out))

        return norm3_out

class GPT(torch.nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, N = 2, dropout = 0.1, max_len = 25):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_embedding = torch.nn.Embedding(max_len, d_model)
        self.decoder_blocks = torch.nn.ModuleList(
            [DecoderBlock(d_model, num_heads) for _ in range(N)]
        )

        self.linear1 = torch.nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, pos_ids, atten_mask=None):  
        token_emb = self.token_embedding(token_ids)
        pos_emb = self.pos_embedding(pos_ids)
        x = token_emb + pos_emb
        for block in self.decoder_blocks:
            x = block(x, atten_mask)

        output = self.linear1(x)

        return torch.softmax(output, dim =-1)

def tokenize(corpus):
    vocab = {"[PAD]":0, "[UNK]":1, "[BOS]":2, "[EOS]":3}
    for poem in corpus:
        for char in poem:
            vocab.setdefault(char, len(vocab))
    id_to_token = {k:v for v,k in vocab.items()}
    return vocab, id_to_token


class Tokenizer:
    def __init__(self, vocab, id_to_token):
        self.vocab = vocab
        self.id_to_token = id_to_token

    def encode(self, text):
        text_ids = []
        for token in text:
            text_ids.append(self.convert_token_to_id(token))
        return text_ids
    
    def decode(self, ids):
        ids_text = []
        for id in ids:
            ids_text.append(self.convert_id_to_token(id))
        return ids_text

    def convert_id_to_token(self, id):
        id_token = self.id_to_token[id]
        return id_token

    def convert_token_to_id(self, token):
        token_id = self.vocab[token]
        return token_id

def create_masked_ids(seq_len):
    masked_ids = torch.triu(torch.ones(1,seq_len, seq_len), diagonal=1)
    return masked_ids == 0

class MyDataset(Dataset):
    def __init__(self, corpus):
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        poem = self.corpus[idx]

        poem_ids = tokenizer.encode(poem) 
        input_ids = [tokenizer.vocab["[BOS]"]] + poem_ids
        label_ids = poem_ids + [tokenizer.vocab["[EOS]"]]
        masked_ids = create_masked_ids(len(input_ids))
        pos_ids = list(range(len(input_ids)))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "masked_ids": masked_ids,
            "pos_ids": torch.tensor(pos_ids, dtype=torch.long)
        }
    
def run_valid():
    model.eval()
    res = []
    with torch.no_grad():
        for i in range(25):
            input_ids = [random.randint(0, vocab_size-1)]
            masked_ids = create_masked_ids(len(input_ids))
            pos_ids = torch.tensor(list(range(len(input_ids))), dtype=torch.long)
            model_out = model(torch.tensor(input_ids, dtype=torch.long).unsqueeze(0), pos_ids.unsqueeze(0), masked_ids.unsqueeze(0))
            probilities = torch.softmax(model_out, dim=-1)
            next_id = torch.argmax(model_out[:, -1, :], dim=-1)
            res.append(next_id.item())
            input_ids.append(next_id.item())
            if next_id.item() == tokenizer.vocab["[EOS]"]:
                break
    return res


if __name__ == "__main__":
    file = "day19/data/5poems3.csv"

    corpus = read_data(file)
    vocab, id_to_token = tokenize(corpus)
    tokenizer = Tokenizer(vocab, id_to_token)
    
    batch_size = 16
    lr = 1e-4
    d_model = 512
    num_heads = 8
    N = 4
    dropout = 0.1
    max_len = 25
    vocab_size = len(vocab)
    epochs = 10

    trainset = MyDataset(corpus)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    model = GPT(vocab_size, d_model, num_heads, N, dropout, max_len)
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        for batch in tqdm(trainloader):
            input_ids = batch["input_ids"]
            label_ids = batch["label_ids"]
            masked_ids = batch["masked_ids"]
            pos_ids = batch["pos_ids"]

            model_out = model(input_ids, pos_ids, masked_ids)
            loss = loss_fn(model_out.reshape(-1, vocab_size), label_ids.reshape(-1))
            loss.backward()
            optim.step()
            optim.zero_grad()

        print(f"e : {e}, loss : {loss}")
        res_ids = run_valid()
        res = tokenizer.decode(res_ids)
        print("生成的古诗：" + "".join(res))
    pass