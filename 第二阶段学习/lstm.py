'''
    lstm的第一次复现
'''
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

def read_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read().split("\n")
    texts = []
    labels = []
    pattern = re.compile(r"^(.*)\s+(\d+)$")
    for line in data:
        matches = pattern.match(line)
        if matches:
            text, label = matches.groups()
            texts.append(text)
            labels.append(int(label))
    return texts, labels

class MyDataset(Dataset):
    def __init__(self, texts, labels, num = None):
        self.texts = texts
        self.labels = labels
        self.num = num
    def __len__(self):
        if self.num:
            return self.num
        else:
            return len(self.texts)

    def __getitem__(self, idx):     # 这里返回的是token的id和label从0的映射
        text = self.texts[idx]
        label = self.labels[idx]
        token_ids = tokenizer.encode_plus(
            text,
            max_length=32,
            truncation=True,
            return_tensors="pt",    # pt：PyTorch's torch.Tensor, tf: TensorFlow's tf.Tensor, np:Numpy's np.ndarray
            add_special_tokens=False,
            padding="max_length"
        )
        return token_ids["input_ids"].squeeze(0), label

def mapping_label(labels):
    mapping = {3:0, 6:1, 7:2, 8:3}
    mapping_labels = [mapping[label] for label in labels]
    return mapping_labels

class LSTM(torch.nn.Module):
    def __init__(self, hidding_num):
        super().__init__()
        self.W = torch.nn.Linear(hidding_num, hidding_num)
        self.U = torch.nn.Linear(hidding_num, hidding_num)
        self.V = torch.nn.Linear(hidding_num, hidding_num)
        # self.relu = torch.nn.ReLU()

    def forward(self, emb):     # emb : batch_size * seq_len * emb_dim
        results = torch.zeros(emb.shape)    # results : batch_size * seq_len * emb_dim
        p1 = torch.zeros(size = (emb.shape[0], emb.shape[2]))      # p1 : batch_size * emb_dim
        p2 = torch.zeros(size = (emb.shape[0], emb.shape[2]))      # p2 : batch_size * emb_dim
        for i in range(seq_len):    # 每个字遍历
            xi = emb[:, i , :]
            xi = xi + p2

            Wxi = self.W(xi)
            Wxi_sigmoid = torch.sigmoid(Wxi)
            p1 = Wxi_sigmoid * p1

            Uxi = self.U(xi)
            Uxi_sigmoid = torch.sigmoid(Uxi)

            Vxi = self.V(xi)
            Vxi_tanh = torch.tanh(Vxi)

            UVxi = Uxi_sigmoid * Vxi_tanh

            p1 = p1 + UVxi

            p2 = torch.sigmoid(xi) * torch.tanh(p1)
            results[:, i, :] = p2
            
        return results

class Model(torch.nn.Module):
    def __init__(self, deepseek_embeddings):
        super().__init__()
        self.embedding = deepseek_embeddings
        self.rnn = LSTM(emb_dim)
        self.classify = torch.nn.Linear(emb_dim, 4)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, batch_token_ids, batch_labels = None):
        # batch_token_ids : batch_size * seq_len
        emb = self.embedding(batch_token_ids)   # emb : batch_size * seq_len * emb_dim
        rnn_out = self.rnn(emb) # rnn_out: batch_size * seq_len * emb_dim
        rnn_out_mean = torch.mean(rnn_out, dim = 1)  #  batch_size * emb_dim
        predicts = self.classify(rnn_out_mean)      # batch_size * label_nums
        if batch_labels is not None:
            loss = self.loss_fn(predicts, batch_labels)
            return loss
        else:
            return predicts

def run_valid():
    model.eval()
    with torch.no_grad():
        current_num = 0
        whole_num = len(testset)
        for batch_token_ids, batch_label in tqdm(testloader):
            predicts = model(batch_token_ids)
            predicts_id = torch.argmax(predicts, dim=1)
            current_num += int(torch.sum(predicts_id == batch_label))
    acc = current_num/whole_num
    print(f"acc: {acc}")

if __name__ == "__main__":
    train_file = "data/train2.txt"
    test_file = "data/test2.txt"
    tokenizer = AutoTokenizer.from_pretrained("model/DeepSeek-R1-Distill-Qwen-1.5B")
    deepseek_model = AutoModel.from_pretrained("model/DeepSeek-R1-Distill-Qwen-1.5B")
    deepseek_embeddings = deepseek_model.get_input_embeddings() # 这里返回的是一个层，也就是torch.nn.embedding, deepseek_embeddings.weight才是tensor类型
    emb_dim = deepseek_embeddings.embedding_dim
    seq_len = 32

    train_texts, train_labels = read_data(train_file)
    test_texts, test_labels = read_data(test_file)

    train_labels = mapping_label(train_labels)
    test_labels = mapping_label(test_labels)

    trainset = MyDataset(train_texts, train_labels,640)
    testset = MyDataset(test_texts, test_labels,64 )

    batch_size = 8
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    lr = 0.001
    epochs = 10
    model = Model(deepseek_embeddings)
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    model.train()
    for epoch in range(epochs):
        for batch_token_ids, batch_label in tqdm(trainloader):
            loss = model(batch_token_ids, batch_label)
            loss.backward()
            optim.step()
            optim.zero_grad()

        print(f"epoch {epoch}, loss: {loss}")   # 这里的Loss是最后一轮的loss
        run_valid()
    pass
