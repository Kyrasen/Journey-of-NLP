import torch
from torch.utils.data import DataLoader, Dataset
import random


def read_data(path):
    with open(path, encoding="utf-8") as f:
        all_data = f.read().split("\n")
    new_data = []

    for data in all_data:
        new_data.append(data[:24])
    return new_data


def build_word_2_index(all_peotry):
    word_2_index = {"U": 0, "E": 1, "，": 2, "。": 3}

    for peotry in all_peotry:
        for w in peotry:
            if w not in word_2_index:
                word_2_index[w] = len(word_2_index)

    index_2_word = list(word_2_index)
    return word_2_index, index_2_word


class MyDataset(Dataset):
    def __init__(self, all_poetry):
        self.all_peotry = all_poetry

    def __len__(self):
        return len(self.all_peotry)

    def __getitem__(self, index):
        text = self.all_peotry[index]
        text_id = [word_2_index.get(i, 0) for i in text]
        label = text_id[1:] + [word_2_index["E"]]

        return torch.tensor(text_id, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(len(word_2_index), emb_dim)
        self.rnn = torch.nn.LSTM(emb_dim, emb_dim, batch_first=True)
        # self.cls = torch.nn.Linear(200,len(word_2_index))
        self.loss_fun = torch.nn.CrossEntropyLoss()

    def forward(self, x, label=None):

        x = self.embedding(x)
        print("x.shape: " , x.shape)
        x, _ = self.rnn(x)
        x = x @ self.embedding.weight.T  # batch * 24 * len(word_2_index)

        if label is not None:
            loss = self.loss_fun(x.reshape(-1, len(word_2_index)), label.reshape(-1))
            return loss
        else:
            return torch.argmax(x, dim=-1)


def random_generate_poetry(text):
    for i in range(24):
        text_id = torch.tensor([[word_2_index[i] for i in text]], device=device)

        predict_idx = int(model.forward(text_id)[0][-1])

        text += index_2_word[predict_idx]

    return text


def hide_head_peotry(input_text):  # 恭喜发财
    result = ""

    for i in range(4):
        t = input_text[i]
        for j in range(4):
            text_id = torch.tensor(
                [[word_2_index.get(i, random.choice([k for k in range(5, len(word_2_index))])) for i in t]],
                device=device)
            predict_idx = int(model.forward(text_id)[0][-1])
            t += index_2_word[predict_idx]
        if i % 2 == 0:
            t += "，"
        else:
            t += "。"
        result += t
    return result


if __name__ == "__main__":
    file = "D:/Desktop/deepLearnProject/day 19/data/5poems2.csv"
    poems = read_data(file)
    all_peotry = poems[:100]

    word_2_index, index_2_word = build_word_2_index(all_peotry)

    batch_size = 1
    lr = 2e-4  # 0.001
    epoch = 20
    emb_dim = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = MyDataset(all_peotry)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 命令变参：*args，未命令变参：**argc

    model = Model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        for batch_text_id, batch_label_id in train_dataloader:
            batch_text_id = batch_text_id.to(device)
            batch_label_id = batch_label_id.to(device)

            loss = model.forward(batch_text_id, batch_label_id)

            loss.backward()
            opt.step()
            opt.zero_grad()
        # print(loss)

        peotry = random_generate_poetry(random.choice(index_2_word[4:]))

        print(peotry, loss)

    while True:
        input_text = input("请输入1个字：")
        if len(input_text) == 1:
            peotry = random_generate_poetry(input_text)
        elif len(input_text) == 4:
            peotry = hide_head_peotry(input_text)
        else:
            peotry = random_generate_poetry(random.choice(index_2_word[4:]))

        print(peotry)