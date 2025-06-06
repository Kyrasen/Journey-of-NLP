import pandas as pd
from anyio import sleep
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import torch
import random

def read_data(file):
    data = pd.read_csv(file)["内容"]
    data_list = list(data)

    return data_list

def tokenizer(data):
    word_to_idx = {"E": 0, "U": 1}

    for poem in data:
        for word in poem:
            word_to_idx.setdefault(word, len(word_to_idx))
    idx_to_word = {k:v for v, k in word_to_idx.items()}
    return word_to_idx, idx_to_word

class MyDataset(Dataset):
    def __init__(self, data, num = None):   # 这里的data是整首诗
        self.data = data
        self.num = num
    def __len__(self):
        if self.num:
            return self.num
        return len(self.data)
    def __getitem__(self, idx):
        poem = data[idx]
        text_ids = [word_to_idx.get(word, word_to_idx["U"]) for word in poem]
        label_ids = text_ids[1:] + [word_to_idx["E"]]

        return torch.tensor(text_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)

class Model(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, emb_dim)
        self.LSTM = torch.nn.LSTM(emb_dim,emb_dim, batch_first=True)

        # self.classcify = self.embeddings.weight.data.T
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, batch_text_ids, batch_label_ids = None):
        emb = self.embeddings(batch_text_ids)
        lstm_out, _ = self.LSTM(emb)
        cls_out = lstm_out @ self.embeddings.weight.data.T

        if batch_label_ids is not None:
            loss = self.loss_fn(cls_out.reshape(-1, vocab_size), batch_label_ids.reshape(-1))
            return loss
        else:
            return torch.argsort(cls_out, dim=-1)    # 也可以取前5个，也可以使用加权法来利用概率来取, torch.multinomial
def generate_poem(head_word):
    for i in range(30):
        with torch.no_grad():
            head_word_id = torch.tensor([[word_to_idx[i] for i in head_word]])
            next_word_id = model(head_word_id)[0,-1].item()
            head_word += idx_to_word[next_word_id]
    return head_word

def hide_head_poem(text):
    result = []
    with torch.no_grad():
        for head in text:
            for i in range(4):
                head_id = torch.tensor([[word_to_idx.get(word, word_to_idx["U"]) for word in head]], dtype=torch.long)
                topk_values, topk_indices = torch.topk(model(head_id),k = 5, dim=-1)    # 这个函数会自动从大到小排序的，k是指去前五个
                next_word_id = random.choice(topk_indices[0][0])
                head += idx_to_word[int(next_word_id)]
            result.append(head)
    return result
if __name__ == "__main__":
    file = "../data/5poems3.csv"    # 使用os.getcwd()确定文件所指的目录是什么

    data = read_data(file)

    word_to_idx, idx_to_word = tokenizer(data)

    epochs = 1
    lr = 1e-3   # 科学计数法
    batch_size = 32
    emb_dim = 1024
    vocab_size = len(word_to_idx)

    trainset = MyDataset(data, 1000)
    trainloader = DataLoader(trainset, batch_size= batch_size, shuffle=True)
    model = Model(emb_dim)
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    for e in range(epochs):
        for batch_text_ids, batch_label_ids in tqdm(trainloader, desc = "trianing ... "):
            loss = model(batch_text_ids, batch_label_ids)

            loss.backward() # 产生梯度
            optim.step()    # 应用梯度
            optim.zero_grad()   # 梯度归零
        print(f"loss : {loss}")

    while(True):
        # head = random.choice(list(idx_to_word.values()))
        # poem = generate_poem(head)
        input_text = input("请输入4个字:")
        hide_head_poems = hide_head_poem(input_text)
        print("".join(hide_head_poems))
    pass