import re
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.read().split("\n")

    inputs = []
    labels = []
    pattern = re.compile(r"^(.*?)\s+(\d+)$")
    for line in data:
        matches = pattern.match(line)
        if matches:
            input, label = matches.groups()
            inputs.append(input)
            labels.append(int(label))
    return inputs, labels

def tokenizer(inputs):
    word_2_idx = {"PAD":0, "UNK":1}
    for sentence in inputs:
        for word in sentence:
            word_2_idx.setdefault(word, len(word_2_idx))

    return word_2_idx

def label_mapping(labels):
    mapping = {3:0, 6:1, 7:2, 8:3}
    new_labels = []

    for label in labels:
        new_labels.append(mapping[label])
    return new_labels

class MyDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]

        input_ids = []
        for token in input:
            input_ids.append(word_2_idx.get(token, word_2_idx["UNK"]))

        return input_ids, label
    
def my_collate_fn(batch_data):
    batch_input_ids, batch_labels = zip(*batch_data)
    new_batch_input_ids = []
    for input_ids in batch_input_ids:
        if len(input_ids) < seq_len:
            input_ids += [word_2_idx["PAD"]] * (seq_len - len(input_ids))
        else:
            input_ids = input_ids[: seq_len]
        new_batch_input_ids.append(input_ids)
    return torch.tensor(new_batch_input_ids, dtype=torch.long), torch.tensor(batch_labels, dtype=torch.long)

class Model(torch.nn.Module):
    def __init__(self, vocab_size,input_emb_dim, hidden_size, label_nums):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size,input_emb_dim)
        self.rnn = torch.nn.LSTM(input_emb_dim, hidden_size, batch_first = True)
        self.cls = torch.nn.Linear(hidden_size,label_nums )

    def forward(self, batch_input_ids):
        input_emb = self.emb(batch_input_ids)   # batch_size * seq_len * emb_dim
        '''
            pytorch里的RNN返回的两个参数：
            output: 所有时间步的隐藏层， shape: batch_size , seq_len , hidden_size
            last_hidden: 最后时刻的隐藏层 ，shape: num_layer(RNN的层数)*bidirectional(双向), batch_size, hidden_size
        '''
        _, (h_t, c_t) = self.rnn(input_emb)    # 1 * batch_size * hidden_size
        cls = self.cls(h_t).squeeze(0)             # 1 * batch_size * label_nums
        return  cls

def run_valid(testloader, testset):
    current_nums = 0
    all_nums = len(testset)
    model.eval()
    with torch.no_grad():
        for batch_input_ids, batch_labels in tqdm(testloader):
            cls = model(batch_input_ids)
            batch_labels = batch_labels.tolist()
            pre_ids = torch.argmax(cls, dim=-1).tolist()
            for pre_id, batch_label in zip(pre_ids, batch_labels):
                if pre_id == batch_label:
                    current_nums += 1
        acc = current_nums / all_nums
    return acc

if __name__ == "__main__":
    train_file = "day10/data/train2.txt"
    test_file = "day10/data/test2.txt"

    '''
        set(train_labels): 3,7,6,8
    '''
    train_inputs, train_labels = read_data(train_file)
    test_inputs, test_labels = read_data(test_file)

    word_2_idx = tokenizer(train_inputs)
    idx_2_word = {k:v for v, k in word_2_idx.items()}
    label_nums = 4
    seq_len = 10

    batch_size = 8
    epochs = 100
    lr = 6e-4
    vocab_size = len(word_2_idx)
    input_emb_dim = 128
    hidden_size = 128

    train_labels = label_mapping(train_labels)
    test_labels = label_mapping(test_labels)

    trainset = MyDataset(train_inputs, train_labels)
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True, collate_fn = my_collate_fn)

    testset = MyDataset(test_inputs, test_labels)
    testloader = DataLoader(testset, batch_size = 2, shuffle=True, collate_fn = my_collate_fn)

    model = Model(vocab_size,input_emb_dim, hidden_size, label_nums)
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        for batch_input_ids, batch_label_ids in tqdm(trainloader, desc = "Training ..."):
            cls = model(batch_input_ids)
            loss = loss_fn(cls, batch_label_ids)
            loss.backward()
            optim.step()
            optim.zero_grad()
        if e % 10 == 0:
            acc = run_valid(testloader, testset)
            print(f"e : {e}, loss : {loss}, acc: {acc}")

    pass
