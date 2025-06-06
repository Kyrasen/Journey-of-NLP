'''
    英译中：
    古诗生成的思路：
    input:  Hi. <BOS>你好
    label:  你好。<EOS>
    
    编解码结构：
    encoder: 用来提取原来的语言的信息
    decoder: 用来提取翻译的语言的信息
    cls    : 用来预测信息的
    loss   : 用来比对实际label和预测label的差，如果用古诗生成的思路的话，放入loss函数中的两个参数需要是一前一后，千万要记住不要弄错了，然后思路不要乱掉，否则会很难受的
'''

import re
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import random

def read_data(file):

    with open(file, "r", encoding="utf-8") as f:
        data = f.read().split("\n")

    pattern = re.compile(r"^(.*?)\t(.*?)\t")
    all_eng = []
    all_chn = []
    for line in data:
        matches = pattern.match(line)
        if matches:
            eng, chn = matches.groups()
            all_eng.append(eng)
            all_chn.append(chn)

    return all_eng[:5000], all_chn[:5000]

def tokenizer(texts):
    vocab = {"PAD":0, "UNK":1, "BOS":2, "EOS":3}
    for text in texts:
        for word in text:
            vocab.setdefault(word, len(vocab))
    return vocab


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.op_vocab = {k:v for v, k in self.vocab.items()}

    def encode(self, text):
        ids = []
        for word in text:
            ids.append(self.convert_word_2_id(word))
        return ids

    def decode(self, ids):
        words = []
        for id in ids:
            words.append(self.convert_id_2_word(id))
        return words

    def convert_word_2_id(self, word):
        id = self.vocab.get(word, self.vocab["UNK"])
        return id
    
    def convert_id_2_word(self, id):
        word = self.op_vocab[id]
        return word


class Encoder(torch.nn.Module):
    def __init__(self,eng_vocab_size, eng_dim, hidden_size):
        super().__init__()
        self.eng_emb = torch.nn.Embedding(eng_vocab_size, eng_dim)
        self.rnn = torch.nn.GRU(eng_dim, hidden_size, batch_first = True)

    def forward(self, eng_ids):
        eng_emb = self.eng_emb(eng_ids)
        _, last_hidden = self.rnn(eng_emb)  # 1, batch_size, hidden_size
        return last_hidden
    
class Decoder(torch.nn.Module):
    def __init__(self, chn_vocab_size, chn_dim, hidden_size):
        super().__init__()
        self.chn_emb = torch.nn.Embedding(chn_vocab_size, chn_dim)
        self.rnn = torch.nn.GRU(chn_dim, hidden_size, batch_first = True)

    def forward(self, eng_info, chn_ids = None):
        chn_emb = self.chn_emb(chn_ids)
        rnn_out, last_hidden = self.rnn(chn_emb, eng_info)
        return rnn_out, last_hidden  # (batch_size, seq_len, hidden_size)

class Model(torch.nn.Module):
    def __init__(self,eng_vocab_size, chn_vocab_size, hidden_size):
        super().__init__()

        self.encoder = Encoder(eng_vocab_size, eng_dim, hidden_size)
        self.decoder = Decoder(chn_vocab_size, chn_dim, hidden_size)
        self.cls = torch.nn.Linear(hidden_size, chn_vocab_size)

    def forward(self, batch_input_ids, batch_labels_ids= None):
        eng_info = self.encoder(batch_input_ids)    # (1, batch_size, hidden_size)
        chn_info, _ = self.decoder(eng_info, batch_labels_ids)

        similarities = self.cls(chn_info)   # (batch_size, seq_len, chn_vocab_size)

        return similarities

class MyDataset(Dataset):
    def __init__(self, all_eng, all_chn, tokenizer_eng, tokenizer_chn):
        self.all_eng = all_eng
        self.all_chn = all_chn
        self.tokenizer_eng = tokenizer_eng
        self.tokenizer_chn = tokenizer_chn

    def __len__(self):
        return len(self.all_eng)
    
    def __getitem__(self, idx):
        input_eng = self.all_eng[idx]
        label_chn = self.all_chn[idx]

        eng_ids = self.tokenizer_eng.encode(input_eng)
        chn_ids = self.tokenizer_chn.encode(label_chn)

        return eng_ids, chn_ids, len(eng_ids), len(chn_ids)
    
def my_collate_fn(batch_data):
    eng_ids, chn_ids, eng_len, chn_len = zip(*batch_data)

    max_eng_len = max(eng_len) 
    max_chn_len = max(chn_len) + 2

    new_eng_ids = []
    new_chn_ids = []

    # 给中文补齐长度
    for chn_id in chn_ids:
        chn_id = [chn_word_2_idx["BOS"]] + chn_id + [chn_word_2_idx["EOS"]]
        chn_id += [chn_word_2_idx["PAD"]] * (max_chn_len - len(chn_id))
        new_chn_ids.append(chn_id)

    # 给英文补齐长度
    for eng_id in eng_ids:
        eng_id += [eng_word_2_idx["PAD"]] * (max_eng_len - len(eng_id))
        
        new_eng_ids.append(eng_id)

    return torch.tensor(new_eng_ids, dtype = torch.long), torch.tensor(new_chn_ids, dtype=torch.long)

def translate(text):
    ids = [chn_word_2_idx["BOS"]]
    seq_len = 10
    model.eval()
    with torch.no_grad():
        text_ids = [eng_word_2_idx.get(word, eng_word_2_idx["UNK"]) for word in text]
        text_ids = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0)
        last_hidden = model.encoder(text_ids)
        input_token = torch.tensor([[chn_word_2_idx["BOS"]]], dtype = torch.long)

        for i in range(seq_len):
            decoder_out, last_hidden = model.decoder(last_hidden, input_token)
            similarities = model.cls(decoder_out)
            pro_id = torch.argmax(similarities[:,-1,:], dim = -1)
            if pro_id.item() == chn_word_2_idx["EOS"]:
                break
            ids.append(pro_id.item())
            input_token = pro_id.unsqueeze(0)
    return ids

if __name__ == "__main__":
    file = "day22/data/cmn.txt"

    all_eng, all_chn = read_data(file)

    eng_word_2_idx = tokenizer(all_eng)
    chn_word_2_idx = tokenizer(all_chn)
    eng_vocab_size = len(eng_word_2_idx)
    chn_vocab_size = len(chn_word_2_idx)

    tokenizer_eng = Tokenizer(eng_word_2_idx)
    tokenizer_chn = Tokenizer(chn_word_2_idx)

    # 调参区
    batch_size = 32
    epochs = 50
    lr = 5e-4
    eng_dim = 256
    chn_dim = 256
    hidden_size = 512

    trainset = MyDataset(all_eng, all_chn,tokenizer_eng, tokenizer_chn)
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True, collate_fn = my_collate_fn)\
    
    model = Model(eng_vocab_size, chn_vocab_size, hidden_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    for e in range(epochs):
        for batch_input_ids, batch_labels_ids in tqdm(trainloader):
            decoder_input = batch_labels_ids[:, :-1]
            target_output = batch_labels_ids[:, 1:]

            similarities = model(batch_input_ids, decoder_input)    # 在预测的时候，使用除了最后一个字的句子
            loss = loss_fn(similarities.reshape(-1, chn_vocab_size), target_output.reshape(-1)) # 在算Loss的时候，使用除了第一个字的句子
            loss.backward()
            optim.step()
            optim.zero_grad()
        print(f"e : {e}, loss : {loss}")

    while True:
        eng_text = random.choice(all_eng)
        translating_ids = translate(eng_text)
        result = tokenizer_chn.decode(translating_ids)
        print(f"eng:{eng_text}, result: {"".join(result)}")

    pass