from torch.utils.data import Dataset, DataLoader
import torch
import math
import re
import random
from transformers import AutoTokenizer, AutoModel

def read_data(file):

    with open(file, "r", encoding="utf-8") as f:
        data = f.read().replace("\u3000", "").strip()
        data = re.split(r"[，。！？；：？\n\t]+", data)

    return data


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

def random_exclude(min, max, exclude):
    while True:
        num = random.randint(min, max)
        if num != exclude:
            break

    return num

def create_masked_ids(ids):
    masked_ids = ids.copy()
    masked_label = [-100] * len(masked_ids)
    for idx,char_id in enumerate(ids):
            if random.random() < 0.15:
                if random.random() < 0.8:
                    masked_ids[idx] = tokenizer_zh.vocab["[MASK]"]
                    masked_label[idx] = char_id
                if random.random() < 0.9:   # 前面0-0.8已经被上面的截断了
                    masked_ids[idx] = random_exclude(0, len(tokenizer_zh.vocab), char_id)
                    masked_label[idx] = char_id

    return masked_ids, masked_label

def pad_to_length(seq_ids, seq_len, padding_id):
    seq_ids = seq_ids[:seq_len]
    seq_ids = seq_ids + [padding_id] * max(0, seq_len-len(seq_ids))

    return seq_ids

def build_dataset(corpus, seq_len):
    new_dataset = []

    for idx,sentence in enumerate(corpus[:-1]):
        current_text = sentence
        next_text = corpus[idx+1]

        rand_seed = random.randint(0,len(corpus)-1)
        while abs(idx - rand_seed) <= 1:
            rand_seed = random.randint(0,len(corpus))

        neg_text = corpus[rand_seed]
        

        new_dataset.append([current_text, next_text, 1])
        new_dataset.append([current_text, neg_text, 0])

    return new_dataset

def CausalMask(seq_len):
    mask = torch.triu(torch.ones(size = (1,seq_len,seq_len)), diagonal = 1)
    return mask == 0

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.seq_len = 25

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        current_text = self.dataset[idx][0]
        next_text = self.dataset[idx][1]

        current_ids = tokenizer_zh.encode(current_text, add_special_tokens=False)
        next_ids = tokenizer_zh.encode(next_text, add_special_tokens=False)

        token_ids = [tokenizer_zh.vocab["[CLS]"]] + current_ids + [tokenizer_zh.vocab["[SEP]"]] + next_ids + [tokenizer_zh.vocab["[SEP]"]]

        seq_ids = [0] *( len(current_ids) + 2) + [1] * (len(next_ids) + 1)

        pe_ids = list(range(0, len(token_ids)))

        masked_ids, masked_label = create_masked_ids(token_ids)
        

        # 统一长度
        token_ids = pad_to_length(masked_ids, seq_len, tokenizer_zh.vocab["[PAD]"])
        seq_ids = pad_to_length(seq_ids, seq_len, tokenizer_zh.vocab["[PAD]"])  # 这里的A句和padding使用的是同一个id
       
        pe_ids = pad_to_length(pe_ids, seq_len, tokenizer_zh.vocab["[PAD]"])

        # masked_ids = pad_to_length(masked_ids, seq_len, tokenizer_zh.vocab["[PAD]"]) 

        masked_label = pad_to_length(masked_label, seq_len, tokenizer_zh.vocab["[PAD]"])
        atten_mask = CausalMask(len(token_ids))
        
        return {
            "token_ids":torch.tensor(token_ids, dtype=torch.long),
            "seq_ids":torch.tensor(seq_ids, dtype=torch.long),
            "atten_mask":atten_mask,
            "masked_label":torch.tensor(masked_label, dtype=torch.long),
            "pos_ids":torch.tensor(pe_ids, dtype=torch.long)
        }
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)

        self.o_proj = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, atten_mask = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(batch_size, self.num_heads, seq_len, self.d_k)
        K = K.view(batch_size, self.num_heads, seq_len, self.d_k)
        V = V.view(batch_size, self.num_heads, seq_len, self.d_k)

        atten_term = Q @ K.transpose(-2,-1) / math.sqrt(self.d_k)
        if atten_mask is not None:
            atten_term.masked_fill_(atten_mask == 0, -1e9)

        atten_scores = torch.softmax(atten_term, dim=-1) @ V
        output = self.o_proj(atten_scores.contiguous().view(batch_size, seq_len, d_model))

        return self.dropout(output)
        
class FeedForward(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        dff = 4 * d_model
        self.W1 = torch.nn.Linear(d_model, dff)
        self.W2 = torch.nn.Linear(dff, d_model)
        self.act = torch.nn.GELU()

    def forward(self,x):
        x = self.W1(x)
        x = self.act(x)
        x = self.W2(x)
        return x
    
class TransformerEncodeBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.lay = torch.nn.LayerNorm(d_model)

    def forward(self,x, atten_mask=None):
        x = self.lay(x + self.mha(x, atten_mask))
        x = self.lay(x + self.ffn(x))
        return x

'''
    token_ids: 一句话中的token_ids（上句 + 下句）
    seq_ids  : 上句的是0，下句的是1
    pos_ids  : 位置的ids，这里是可学习的，与transformer的固定不一样
'''
class Bert(torch.nn.Module):
    def __init__(self,vocab_size, d_model, seq_len, num_heads,dropout=0.1):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.seq_embedding = torch.nn.Embedding(2, d_model)
        self.pe_embedding = torch.nn.Embedding(100, d_model)
        self.encoder = TransformerEncodeBlock(d_model, num_heads)

        self.cls = torch.nn.Linear(d_model, vocab_size)


    def forward(self, token_ids, seq_ids, pos_ids, atten_mask=None):
        token_embs = self.token_embedding(token_ids)
        seg_embs = self.seq_embedding(seq_ids)
        pos_embs = self.pe_embedding(pos_ids)
        total_embs = token_embs + seg_embs + pos_embs
        encoder_output = self.encoder(total_embs, atten_mask)
        output = self.cls(encoder_output)

        return output

def run_valid():
    model.eval()
    with torch.no_grad():
        current_sentence = corpus[random.randint(0, len(corpus)-1)]
        current_token_ids = torch.tensor([tokenizer_zh.vocab["[CLS]"]] + tokenizer_zh.encode(current_sentence) + [tokenizer_zh.vocab["[SEP]"]], dtype=torch.long).unsqueeze(0)
        
        res = []
        for i in range(50):
            seq_len = current_token_ids.shape[1]
            current_seq_ids = torch.zeros_like(current_token_ids)
            current_pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

            model_out = model(current_token_ids, current_seq_ids, current_pos_ids)

            next_token_logits = model_out[:, -1, :]
            
            temperature = 1.0
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples = 1)

            if next_id.item() == tokenizer_zh.vocab["[SEP]"]:
                break
            res.append(next_id.item())
            current_token_ids = torch.cat(
                (current_token_ids,
                next_id
                ),
                dim = 1
            )
        res_sentence = tokenizer_zh.decode(res)
        print(f"src: {current_sentence}, generate : {"".join(res_sentence)}")
        
        
if __name__ == "__main__":

    file = "day26/data/白杨礼赞.txt"
    stop_words = ["，", "。", "！", "；", "：", "\n", "\t", "？"]
    model_path = "models/bert-base-chinese"
    
    corpus = read_data(file)
    seq_len = 30
    # data = build_dataset(corpus, stop_words)
    tokenizer_zh = AutoTokenizer.from_pretrained(model_path, trust_remote_mode = True, local_files_only=True)    # 可以用别人预训练好的模型进行分词， tokenizer -> model

    dataset = build_dataset(corpus,seq_len)
    seq_len = 15
    d_model = 768
    dropout = 0.1
    epochs = 10
    num_heads = 8
    lr = 0.001

    trainset = MyDataset(dataset)
    trainloader = DataLoader(trainset, batch_size = 8)

    model = Bert(len(tokenizer_zh.vocab), d_model, seq_len, num_heads, dropout)
    optim = torch.optim.Adam(model.parameters(),lr = lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = -100)

    for e in range(epochs):
        for batch in trainloader:
            token_ids = batch["token_ids"]
            seq_ids = batch["seq_ids"]
            pos_ids = batch["pos_ids"]
            atten_mask = batch["atten_mask"]
            masked_label = batch["masked_label"]

            model_out = model(token_ids, seq_ids, pos_ids, atten_mask)
            loss = loss_fn(model_out.view(-1, len(tokenizer_zh.vocab)), masked_label.view(-1))

            loss.backward()
            optim.step()
            optim.zero_grad()

        run_valid()
        print(f"e : {e}, loss: {loss}")
    
    pass 