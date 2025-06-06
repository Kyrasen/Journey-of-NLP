import re
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch

def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.read().split("\n")

    pattern = re.compile(r"^(.*?)\t(.*?)\t")
    all_eng1 = []
    all_chn1 = []
    for line in data:
        matches = pattern.match(line)
        if matches:
            eng, chn = matches.groups()
            all_eng1.append(eng)
            all_chn1.append(chn)
    return all_chn1[:500], all_eng1[:500]

def tokenizer(all_chn, all_eng):
    '''
        中文给每个字一个id
        英文给每个字母一个id
    '''
    chn_word_2_idx = {"PAD":0, "UNK":1, "BOS":2, "EOS":3}
    eng_word_2_idx = {"PAD":0, "UNK":1, "BOS":2, "EOS":3}

    for line in all_chn:
        for word in line:
            chn_word_2_idx.setdefault(word, len(chn_word_2_idx))

    for line in all_eng:
        for letter in line:
            eng_word_2_idx.setdefault(letter, len(eng_word_2_idx))

    return chn_word_2_idx, eng_word_2_idx

class MyDataset(Dataset):
    def __init__(self, all_chn, all_eng):
        self.all_chn = all_chn
        self.all_eng = all_eng
    def __len__(self):
        return len(self.all_eng)
    def __getitem__(self, idx):
        chn_sentence = self.all_chn[idx]
        eng_sentence = self.all_eng[idx]

        chn_sentence_ids = [chn_word_2_idx.get(word, chn_word_2_idx["UNK"]) for word in chn_sentence]
        eng_sentence_ids = [eng_word_2_idx.get(word, eng_word_2_idx["UNK"]) for word in eng_sentence]

        chn_sentence_len = len(chn_sentence_ids)
        eng_sentence_len = len(eng_sentence_ids)
        return  chn_sentence_ids, eng_sentence_ids, chn_sentence_len, eng_sentence_len

def my_collate_fn(batch_data):
    batch_eng_ids = []
    batch_chn_ids = []
    chn_len1 = []
    eng_len1 = []
    for chn_ids, eng_ids, chn_len, eng_len in batch_data:
        batch_eng_ids.append(eng_ids)
        batch_chn_ids.append(chn_ids)
        chn_len1.append(chn_len)
        eng_len1.append(eng_len)

    max_chn_len = max(chn_len1)
    max_eng_len = max(eng_len1) + 2  # 这里的2是加上了BOS和EOS

    new_batch_chn_ids = []
    new_batch_eng_ids = []
    # 中文句补齐
    for ids in batch_chn_ids:
        padding =ids + [chn_word_2_idx["PAD"]] * (max_chn_len - len(ids))
        new_batch_chn_ids.append(padding)

    # 英文句补齐，同时加上BOS 和 EOS，一定要加，不然模型不知道什么时候结束
    for ids in batch_eng_ids:
        padding = [eng_word_2_idx["BOS"]] + ids + [eng_word_2_idx["EOS"]]
        padding += [eng_word_2_idx["PAD"]] *( max_eng_len - len(padding) )
        new_batch_eng_ids.append(padding)

    return torch.tensor(new_batch_chn_ids, dtype=torch.long), torch.tensor(new_batch_eng_ids, dtype=torch.long)

class Model(torch.nn.Module):
    def __init__(self,chn_emb, eng_emb, hidden_size = 424):
        super().__init__()
        self.chn_embedding = torch.nn.Embedding(chn_vocab_size, chn_emb)    # 中文的词向量
        self.eng_embedding = torch.nn.Embedding(eng_vocab_size, eng_emb)    # 英文的词向量

        self.encoder = torch.nn.GRU(chn_emb, hidden_size, batch_first=True)       # 对中文进行编码
        self.decoder = torch.nn.GRU(eng_emb, hidden_size, batch_first=True)       # 对英文进行解码

        self.cls = torch.nn.Linear(hidden_size, eng_vocab_size)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, batch_chn_ids, batch_eng_ids = None):
        '''
            1、首先，放入encoder和decoder的应该都是词向量，这个词向量由embedding生成
            2、然后搞清楚放入循环神经网路的信息应该是哪个，放入encoder的是中文的词向量，得到的是中文最后一个字的信息
                放入 decoder 的是由encoder得到的中文最后一个字的信息和英文的词向量（除去最后一个字符），得到的是英文全部时间步的信息
            3、将上一步得到的英文全部时间步的信息放入到分类器中，得到分类概率
            4、算loss: 输入：分类概率矩阵、英文句子（减去第一个词）

        '''
        if batch_eng_ids is not None:
            chn_emb = self.chn_embedding(batch_chn_ids)
            _, chn_out = self.encoder(chn_emb)  # 对中文放到循环神经网络当中
            eng_emb = self.eng_embedding(batch_eng_ids[:, :-1])
            eng_out, _ = self.decoder(eng_emb, chn_out)  # 将中文作为历史信息，英文也作为信息放到循环神经网络当中

            cls = self.cls(eng_out)
            loss = self.loss_fn(cls.reshape(-1, eng_vocab_size), batch_eng_ids[:, 1:].reshape(-1))
            return loss
        else:
            result = []
            chn_ids = [chn_word_2_idx.get(word, chn_word_2_idx["UNK"]) for word in input_text]
            chn_ids = torch.tensor(chn_ids, dtype=torch.long).unsqueeze(0)
            chn_emb = self.chn_embedding(chn_ids)
            _, chn_out = self.encoder(chn_emb)
            hidden = chn_out    # 将历史信息保存下来，然后作为每一次的字母的历史信息
            eng_ids = torch.tensor([[ eng_word_2_idx["BOS"] ]], dtype=torch.long)
            for i in range(20):
                eng_emb = self.eng_embedding(eng_ids)

                eng_out, hidden = self.decoder(eng_emb, hidden)
                cls = self.cls(eng_out)
                next_id = torch.argmax(cls[:, -1, :], dim=-1).item()
                if next_id == eng_word_2_idx["EOS"]:
                    break

                result.append(eng_idx_2_word.get(next_id, "UNK"))
                eng_ids = torch.tensor([[next_id]], dtype=torch.long)
            return "".join(result)

if __name__ == "__main__":
    file = "../data/cmn.txt"

    all_chn, all_eng = read_data(file)

    chn_word_2_idx, eng_word_2_idx = tokenizer(all_chn, all_eng)
    eng_idx_2_word = {k: v for v, k in eng_word_2_idx.items()}
    chn_vocab_size = len(chn_word_2_idx)
    eng_vocab_size = len(eng_word_2_idx)

    trainset = MyDataset(all_chn, all_eng)
    trainloader = DataLoader(trainset, batch_size=2, shuffle = True, collate_fn= my_collate_fn)

    chn_emb_size = 100
    eng_emb_size = 100
    lr = 5e-4
    epochs = 20

    model = Model(chn_emb_size, eng_emb_size)
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    for e in range(epochs):
        for batch_chn_ids, batch_eng_ids in tqdm(trainloader, desc="training ..."):
            loss = model(batch_chn_ids, batch_eng_ids)

            loss.backward()
            optim.step()
            optim.zero_grad()
        print(f"epoch : {e}, loss : {loss}")

    print(all_chn[:1000])

    '''
        1、当模型不断重复同一个字母时，说明每次预测的是同一个字，那么就想想推理是放入的历史信息是否改变
        2、当模型学不到结束符时，就想想英文句子是否加入了"EOS"
        3、我们的目的是这样的：输入：(chn) BOS I love you ， 预测目标：I love you EOS，其实这个就是seq2seq的典型的思想
    '''

    while True:
        input_text = input("请输入中文：")
        answer = model(input_text)
        print(answer)
    pass
