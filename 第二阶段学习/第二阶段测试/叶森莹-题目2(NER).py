import torch
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score, accuracy_score

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
        # self.seq_embedding = torch.nn.Embedding(2, d_model)
        self.pe_embedding = torch.nn.Embedding(100, d_model)
        self.encoder = TransformerEncodeBlock(d_model, num_heads)

        self.cls = torch.nn.Linear(d_model, len(label_list))


    # def forward(self, token_ids, seq_ids, pos_ids, atten_mask=None):
    def forward(self, token_ids, pos_ids, atten_mask=None):
        token_embs = self.token_embedding(token_ids)
        # seg_embs = self.seq_embedding(seq_ids)
        pos_embs = self.pe_embedding(pos_ids)
        total_embs = token_embs + pos_embs
        # total_embs = token_embs + seg_embs + pos_embs
        encoder_output = self.encoder(total_embs, atten_mask)
        output = self.cls(encoder_output)

        return output
    
def read_ner_data(file_path):
    texts, tags = [], []
    with open(file_path, encoding='utf-8') as f:
        words, labels = [], []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    texts.append(words)
                    tags.append(labels)
                    words, labels = [], []
                continue
            char, label = line.split()
            words.append(char)
            labels.append(label)
        if words:
            texts.append(words)
            tags.append(labels)
    return texts, tags

def tokenize(train_texts):
    vocab = {"[UNK]":0, "[unused1]":1,"[unused2]":2, "[PAD]":3}
    for sentence in train_texts:
        for char in sentence:
            vocab.setdefault(char, len(vocab))
    idx_to_token = {k:v for v, k in vocab.items()}
    return vocab, idx_to_token

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

def pad_to_length(seq_ids, seq_len, padding_id):
    seq_ids = seq_ids[:seq_len]
    seq_ids = seq_ids + [padding_id] * max(0, seq_len-len(seq_ids))

    return seq_ids

def CausalMask(seq_len):
    mask = torch.triu(torch.ones(size = (1,seq_len,seq_len)), diagonal = 1)
    return mask == 0

class MyDataset(Dataset):
    def __init__(self, corpus, labels):
        self.corpus = corpus
        self.labels = labels

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        texts = self.corpus[idx]
        labels = self.labels[idx]

        token_ids = tokenizer_text.encode(texts)
        seq_ids = [0] * len(token_ids)
        pos_ids = list(range(len(token_ids)))
        labels_ids = tokenizer_label.encode(labels)
        

        # 对齐处理
        token_ids = pad_to_length(token_ids, seq_len, token_to_idx["[PAD]"])
        seq_ids = pad_to_length(seq_ids, seq_len, token_to_idx["[PAD]"])
        pos_ids = pad_to_length(pos_ids, seq_len, token_to_idx["[PAD]"])
        labels_ids = pad_to_length(labels_ids, seq_len, token_to_idx["[PAD]"])
        atten_mask = CausalMask(len(token_ids))
        

        return {
            "token_ids":torch.tensor(token_ids, dtype=torch.long),
            "seq_ids": torch.tensor(seq_ids, dtype=torch.long),
            "pos_ids": torch.tensor(pos_ids, dtype=torch.long),
            "labels_ids": torch.tensor(labels_ids, dtype=torch.long),
            "atten_mask":atten_mask
        }

def run_valid():
    model.eval()
    all_true_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(validloader, desc="validset ..."):
            input_ids = batch["token_ids"]
            pos_ids = batch["pos_ids"]
            labels_ids = batch["labels_ids"]
            atten_mask = batch["atten_mask"]

            model_out = model(input_ids, pos_ids, atten_mask)
            pred_labels = torch.argmax(model_out, dim=-1)

            if isinstance(pred_labels, torch.Tensor):
                pred_labels = pred_labels.cpu().numpy().tolist()
            if isinstance(labels_ids, torch.Tensor):
                labels_ids = labels_ids.cpu().numpy().tolist()

            for true_seq, pred_seq in zip(labels_ids, pred_labels):
                true_seq_labels = []
                pred_seq_labels = []
                for t, p in zip(true_seq, pred_seq):
                    if t == -100:
                        continue
                    true_seq_labels.append(id_to_label[t])
                    pred_seq_labels.append(id_to_label[p])
                all_true_labels.append(true_seq_labels)
                all_preds.append(pred_seq_labels)

    print("ACC: ", accuracy_score(all_true_labels, all_preds))
    print(f"F1 Score: {f1_score(all_true_labels, all_preds):.4f}")
    print(classification_report(all_true_labels, all_preds))


if __name__ == "__main__":
    train_file = "day29Test/叶森莹-题目2/data/train.txt"
    valid_file = "day29Test/叶森莹-题目2/data/train.txt"

    train_texts, train_tags = read_ner_data(train_file)
    valid_texts, valid_tags = read_ner_data(valid_file)

    label_list = sorted(set(tag for doc in train_tags for tag in doc))
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    token_to_idx, idx_to_token = tokenize(train_texts)
    vocab_size = len(token_to_idx)
    tokenizer_text = Tokenizer(token_to_idx, idx_to_token)
    tokenizer_label = Tokenizer(label_to_id, id_to_label)

    seq_len = 30
    batch_size = 32
    lr = 1e-4
    epochs = 10
    d_model = 768
    dropout = 0.1
    num_heads = 12

    trainset = MyDataset(train_texts, train_tags)
    trainloader = DataLoader(trainset, batch_size = batch_size,  shuffle=True)

    validset = MyDataset(valid_texts, valid_tags)
    validloader = DataLoader(validset, batch_size = batch_size,  shuffle=True)

    model = Bert(vocab_size, d_model, seq_len, num_heads, dropout)
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        for batch in tqdm(trainloader, desc = "training ..."):
            input_ids = batch["token_ids"]
            # seq_ids = batch["seq_ids"]
            pos_ids = batch["pos_ids"]
            labels_ids = batch["labels_ids"]
            atten_mask = batch["atten_mask"]

            # model_out = model(input_ids, seq_ids, pos_ids, atten_mask)
            model_out = model(input_ids, pos_ids, atten_mask)
            loss = loss_fn(model_out.reshape(-1, len(label_list)), labels_ids.reshape(-1))

            loss.backward()
            optim.step()
            optim.zero_grad()

        print(f"e : {e}, loss : {loss}")
        run_valid()
            
    pass