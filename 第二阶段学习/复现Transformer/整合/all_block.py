import torch
import math
import os
from torch.utils.data import Dataset, DataLoader
import random

class Encoding_embedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, input_ids):
        embs = self.embedding(input_ids)

        return (embs * math.sqrt(self.d_model))

class Positional_Encoding(torch.nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.d_model = d_model
        self.register_buffer("pe", self._get_pe(seq_len))   # 将pe注册到模型当中，不会进行梯度更新，可保存在state_dict中，适合位置编码或mask这种固定的值

    def _get_pe(self, seq_len):
        # 这些事为了让模型感知位置的
        pos = torch.arange(0, seq_len).unsqueeze(1)
        even = torch.arange(0, self.d_model, 2)
        exp_term = torch.exp((-even * math.log(10000)) / self.d_model)
        term = pos * exp_term   # pos : (seq_len,1), exp_term: (d_model/2),term: (seq_len, d_model/2)

        pe = torch.zeros(size=(1, seq_len, self.d_model))
        pe[:, :, 0::2] = torch.sin(term)
        pe[:, :, 1::2] = torch.cos(term)

        return pe

    def forward(self, x):
        '''
            x: Embedding output, shape = (batch, seq_len, d_model)
        '''
        x_seq_len = x.shape[1]

        if x_seq_len > self.pe.size(1):
            self.pe = self._get_pe(x_seq_len)

        emb_pos = x + self.pe[:, :x_seq_len, :]
        return self.dropout(emb_pos)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads,d_model, dropout = 0.1):
        super().__init__()
        self.softmax = torch.nn.Softmax()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model // self.num_heads
        self.dropout = torch.nn.Dropout(dropout)

        self.q_proj_weights = torch.nn.Linear(self.d_model, self.d_model)
        self.v_proj_weights = torch.nn.Linear(self.d_model, self.d_model)
        self.k_proj_weights = torch.nn.Linear(self.d_model, self.d_model)

        self.o_proj_weights = torch.nn.Linear(self.d_model, self.d_model)

    def attention(self, Q, K, V, atten_mask=None) -> torch.Tensor:
        term = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        
        if atten_mask is not None:
            term.masked_fill_(atten_mask == 0, 1e-9)    # 函数名_最后有个"_"，这种是原地操作函数，直接修改张量本身

        atten_scores = torch.softmax(term , dim=-1) # 这里的torch.softmax是一个函数
        atten_scores = atten_scores @ V
        
        return atten_scores
    
    def forward(self, query, key, value, atten_mask=None):
        # (batch, seq_len, d_model)
        batch_size, seq_len, _ = query.shape
        d_k = int(self.d_model / self.num_heads)
        Q = self.q_proj_weights(query)
        K = self.k_proj_weights(key)
        V = self.v_proj_weights(value)

        Q = Q.reshape(batch_size, self.num_heads, Q.shape[1], d_k)
        K = K.reshape(batch_size, self.num_heads, K.shape[1], d_k)
        V = V.reshape(batch_size, self.num_heads, V.shape[1], d_k)

        atten_scores = self.attention(Q, K, V, atten_mask)
        output = self.o_proj_weights(atten_scores.reshape(batch_size, seq_len, self.d_model))
        
        return self.dropout(output)
    

class LayerNorm(torch.nn.Module):
    def __init__(self, d_model,eps = 1e-5):
        super().__init__()
        self.eps = 1e-5

        self.gamma = torch.nn.Parameter(torch.ones(d_model))    # 类似于斜率
        self.beta = torch.nn.Parameter(torch.zeros(d_model))    # 类似于偏置常数

    def forward(self, inputs_emb):
        mean = torch.mean(inputs_emb, dim=-1, keepdim=True)
        std = torch.std(inputs_emb, dim=-1, keepdim=True)
        output = ((inputs_emb - mean) / (std+self.eps)) * self.gamma + self.beta

        return output
    

class FeedForward(torch.nn.Module):
    def __init__(self, d_model,dff = 2048, dropout=0.1):
        super().__init__()
        self.W1 = torch.nn.Linear(d_model, dff)
        self.W2 = torch.nn.Linear(dff, d_model)
        self.relu = torch.nn.ReLU()

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_emb):
        x = self.W1(input_emb)
        x = self.relu(x)
        x = self.W2(x)
        return self.dropout(x)
    

class Residual(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lan = LayerNorm(d_model)

    def forward(self, x, sublayer):
        res = x + sublayer(x)
        return self.lan(res)
    
class EncoderBlock(torch.nn.Module):
    def __init__(self, num_heads, d_model, dff=2048, dropout=0.1):
        super().__init__()
        self.mha_block = MultiHeadAttention(num_heads, d_model)
        self.ffn_block = FeedForward(d_model, dff, dropout)

        self.residual1 = Residual(d_model)  # multiheadAttention
        self.residual2 = Residual(d_model)  # feedforward

    def forward(self, x, atten_mask = None):
        x = self.residual1(x, lambda x: self.mha_block(x,x,x,atten_mask))
        x = self.residual2(x, self.ffn_block) # 等价于x = self.residual2(x, lambda x: self.ffn_block)
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self,
                d_model,
                dropout,
                num_heads,
                dff, 
                N = 6
                ):
        super().__init__()
        encoder_blocks = [EncoderBlock(num_heads,d_model, dff, dropout) for _ in range(N)]

        self.encoder_blocks = torch.nn.ModuleList(encoder_blocks)

    def forward(self,x,atten_mask = None):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x,atten_mask)
        return x
    
class DecoderBlock(torch.nn.Module):
    def __init__(self,
                 num_heads,
                 d_model,
                 dropout,
                 dff
                ):
        super().__init__()
        self.mha_block = MultiHeadAttention(num_heads, d_model, dropout)
        self.cross_mha_block = MultiHeadAttention(num_heads, d_model, dropout)
        self.ffn_block = FeedForward(d_model, dff, dropout)

        self.residual1 = Residual(d_model)
        self.residual2 = Residual(d_model)
        self.residual3 = Residual(d_model)


    def forward(self, x, encoder_output,decoder_mask=None,encoder_mask=None):
        x = self.residual1(x, lambda x: self.mha_block(x,x,x,decoder_mask))
        x = self.residual2(x, lambda x: self.cross_mha_block(x, encoder_output, encoder_output,encoder_mask))
        x = self.residual3(x, self.ffn_block)

        return x
    
class Decoder(torch.nn.Module):
    def __init__(self,
                 num_heads,
                 d_model,
                 dropout,
                 dff,
                 N = 6
                ):
        super().__init__()
        decoder_blocks = [DecoderBlock(num_heads, d_model, dropout, dff) for _ in range(N)]

        self.decoder_blocks = torch.nn.ModuleList(decoder_blocks)

    def forward(self,x, encoder_output,decoder_mask=None,encoder_mask=None):
        for decoder_block in self.decoder_blocks:
            x = decoder_block( x, encoder_output,decoder_mask,encoder_mask)
        return x
    
class Transformer(torch.nn.Module):
    def __init__(self,
                eg_vocab_size, 
                zh_vocab_size,
                d_model,
                seq_len,
                num_heads,
                dff,
                dropout=0.1
                 
                ):
        super().__init__()
        self.eg_embedding = Encoding_embedding(eg_vocab_size, d_model)
        self.zh_embedding = Encoding_embedding(zh_vocab_size, d_model)

        self.eg_pe = Positional_Encoding(d_model, seq_len, dropout)
        self.zh_pe = Positional_Encoding(d_model, seq_len, dropout)

        self.encoder = Encoder(d_model, dropout, num_heads, dff)
        self.decoder = Decoder(num_heads, d_model, dropout, dff)

        self.proj = torch.nn.Linear(d_model, zh_vocab_size)

    
    def encode(self,x, atten_mask=None):
        x = self.eg_embedding(x)
        x = self.eg_pe(x)
        x = self.encoder(x, atten_mask)
        return x
    
    def decode(self, x, encoder_output,encoder_mask=None ,decoder_mask=None):
        x = self.zh_embedding(x)
        x = self.zh_pe(x)
        x = self.decoder(x, encoder_output, decoder_mask, encoder_mask)
        return x
    
    def project(self,x):
        return self.proj(x)
    
def read_data(file):

    with open(file, "r", encoding="utf-8") as f:
        data = f.read().strip().split("\n")

    return data

    
def tokenize(corpus):
    vocab = {"[UNK]":0, "[PAD]":1, "[BOS]":2, "[EOS]":3}
    for line in corpus:
        for char in line:
            vocab.setdefault(char, len(vocab))
    id_to_token = {k:v for v, k in vocab.items()}

    return vocab, id_to_token

class Tokenizer:
    def __init__(self, vocab, id_to_token):
        self.vocab = vocab
        self.id_to_token = id_to_token

    def encode(self, sentence):
        ids = []
        for char in sentence:
            ids.append(self.convert_token_to_id(char))
        return ids
    
    def decode(self, ids):
        chars = []
        for id in ids:
            chars.append(self.convert_id_to_token(id))
        return chars

    def convert_token_to_id(self, char):
        char_id = self.vocab[char]
        return char_id 
    
    def convert_id_to_token(self, id):
        id_char = self.id_to_token[id]
        return id_char
    
def PaddingMask(x):
    return x != 1

def CausalMask(seq_len):
    mask = torch.triu(torch.ones(size = (1,seq_len,seq_len)), diagonal = 1)
    return mask == 0

    
class MyDataset(Dataset):
    def __init__(self,zh_corpus, eg_corpus, en_tokenizer: Tokenizer, zh_tokenizer: Tokenizer, seq_len=10):
        self.zh_corpus = zh_corpus
        self.eg_corpus = eg_corpus
        self.zh_tokenizer = zh_tokenizer
        self.eg_tokenizer = eg_tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.zh_corpus)

    def __getitem__(self, idx):
        encoder_text = self.eg_corpus[idx]
        decoder_text = self.zh_corpus[idx]

        encoder_ids = self.eg_tokenizer.encode(encoder_text)
        decoder_ids = self.zh_tokenizer.encode(decoder_text)

        encoder_ids = encoder_ids[: self.seq_len-2]
        decoder_ids = decoder_ids[: self.seq_len-1]

        encoder_padding_num = self.seq_len - 2 - len(encoder_ids)
        decoder_padding_num = self.seq_len - 1 - len(decoder_ids)

        encoder_input = torch.cat(
            (
            torch.tensor([self.eg_tokenizer.vocab["[BOS]"]], dtype=torch.long),
            torch.tensor(encoder_ids, dtype=torch.long),
            torch.tensor([self.eg_tokenizer.vocab["[EOS]"]], dtype=torch.long),
            torch.tensor([self.eg_tokenizer.vocab["[PAD]"]] * max(0,encoder_padding_num), dtype=torch.long)
            ),
            dim = 0
        )

        decoder_input = torch.cat(
            (
            torch.tensor([self.eg_tokenizer.vocab["[BOS]"]], dtype=torch.long),
            torch.tensor(decoder_ids, dtype=torch.long),
            torch.tensor([self.eg_tokenizer.vocab["[PAD]"]] * max(0,decoder_padding_num), dtype=torch.long)
            ),
            dim = 0
        )

        label = torch.cat(
            (
            torch.tensor(decoder_ids, dtype=torch.long),
            torch.tensor([self.eg_tokenizer.vocab["[EOS]"]], dtype=torch.long),
            torch.tensor([self.eg_tokenizer.vocab["[PAD]"]] * max(0, decoder_padding_num), dtype=torch.long)
            ),
            dim = 0
        )
        # 最终他们维度需要变成：batch, heads, seq_len, seq_len
        padding_mask = PaddingMask(encoder_input).unsqueeze(0).unsqueeze(0)   # shape : seq_len -> batch, seq_len
        causal_mask = CausalMask(self.seq_len)  # shape : 1, seq_len, seq_len -> batch, 1, seq_len, seq_len
        return {
            "encoder_text":encoder_text,
            "decoder_text":decoder_text,
            "encoder_input":encoder_input,
            "decoder_input":decoder_input,
            "label":label,
            "padding_mask":padding_mask,
            "causal_mask":causal_mask
        }
    
def run_valid(batch):
    model.eval()
    with torch.no_grad():
        encoder_input = batch["encoder_input"].unsqueeze(0)
        encoder_mask = batch["padding_mask"].unsqueeze(0)

        encoder_out = model.encode(encoder_input, encoder_mask)

        decoder_input = torch.tensor([[zh_vocab["[BOS]"]]], dtype=torch.long)
        res = [zh_vocab["[BOS]"]]
        while len(decoder_input) < seq_len:
            t = decoder_input.shape[1]
            decoder_mask = CausalMask(t).unsqueeze(0)
            decoder_out = model.decode(decoder_input, encoder_out, encoder_mask, decoder_mask)
            probilities = model.proj(decoder_out[:, -1, :])
            next_word_id = torch.argmax(probilities, dim = -1)
            if next_word_id.item() == zh_vocab["[EOS]"]:
                break
            res.append(next_word_id.item())
            decoder_input = torch.cat((decoder_input.squeeze(0), next_word_id), dim=-1).unsqueeze(0)

    return res


if __name__ == "__main__":
    zh_file = "day25/data/zh.txt"
    eg_file = "day25/data/eg.txt"

    zh_corpus = read_data(zh_file)
    eg_corpus = read_data(eg_file)

    zh_vocab, zh_id_to_token = tokenize(zh_corpus)
    eg_vocab, eg_id_to_token = tokenize(eg_corpus)

    zh_tokenizer = Tokenizer(zh_vocab, zh_id_to_token)
    eg_tokenizer = Tokenizer(eg_vocab, eg_id_to_token)

    trainset = MyDataset(zh_corpus, eg_corpus, eg_tokenizer, zh_tokenizer)
    trainloader = DataLoader(trainset, batch_size=5, shuffle=True)

    epochs = 20
    lr = 1e-4
    d_model = 512
    seq_len = 10
    num_heads = 8
    dff = 2048
    dropout = 0.1

    model = Transformer(len(eg_vocab), len(zh_vocab), d_model, seq_len, num_heads, dff, dropout)
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        for batch in trainloader:
            encoder_input = batch["encoder_input"]
            decoder_input = batch["decoder_input"]

            encoder_mask = batch["padding_mask"]
            decoder_mask = batch["causal_mask"]

            encoder_out = model.encode(encoder_input, encoder_mask)
            decoder_out = model.decode(decoder_input, encoder_out, encoder_mask, decoder_mask)

            prob = model.proj(decoder_out)
            loss = loss_fn(prob.reshape(-1, len(zh_vocab)), batch["label"].reshape(-1))
            loss.backward()
            optim.step()
            optim.zero_grad()

        print(f"e : {e}, loss: {loss}")

    for i in range(50):
        sample_id = random.randint(0, len(eg_corpus)-1)
        batch = trainset[sample_id]
        res_ids = run_valid(batch)
        res = zh_tokenizer.decode(res_ids)
        print(f"src: {batch['encoder_text']}, res : {"".join(res)}")

            
    pass