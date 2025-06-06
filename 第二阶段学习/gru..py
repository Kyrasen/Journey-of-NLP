'''
    GRU复现：first time
    任务：文本分类
'''
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.read().split("\n")
    pattern = re.compile(r"^(.*)\s+(\d+)$")
    texts = []
    labels = []
    for line in data:
        matches = pattern.match(line)
        if matches:
            text, label = matches.groups()
            texts.append(text)
            labels.append(int(label))
    return texts, labels

def mapping_label(labels):
    mapping = {3:0, 6:1, 7:2, 8:3}
    mapping_labels = []
    for label in labels:
        label = mapping[label]
        mapping_labels.append(label)
    return mapping_labels

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
    def __getitem__(self, idx): # 这里返回的是deepseek的tokens_id和label的映射（因为Loss函数是多元交叉熵）
        text = self.texts[idx]
        label = self.labels[idx]

        tokens_id = deepseek_tokenizer.encode_plus(
            text,
            max_length=32,
            padding="max_length",   # 短了补齐
            truncation=True,        # 长了截断
            add_special_tokens=False,
            return_tensors="pt"     # 返回类型为pytorch的tensor
        )       # encode_plus返回的是字典，我们只需要key为input_ids里面的id
        return tokens_id["input_ids"].squeeze(0), label

class gru(torch.nn.Module):
    def __init__(self, hidden_num):
        super().__init__()
        self.W = torch.nn.Linear(hidden_num, hidden_num)
        self.U = torch.nn.Linear(hidden_num, hidden_num)
        self.V = torch.nn.Linear(hidden_num, hidden_num)

    def forward(self, emb):
        p = torch.zeros(size=(batch_size, emb_dim))
        results = torch.zeros(emb.shape)
        for i in range(seq_len):
            xi = emb[:, i, :]   # xi : batch_size * 1 * emb_dim
            p_xi = xi + p

            W_p_xi = self.W(p_xi)
            W_p_xi_sigmoid = torch.sigmoid(W_p_xi)

            V_xi = self.V(W_p_xi_sigmoid*p + xi)
            V_xi_tanh = torch.tanh((V_xi))

            U_xi = self.U(p_xi)
            U_xi_sigmoid = torch.sigmoid(U_xi)

            mlus1 = 1 - U_xi_sigmoid

            p = mlus1*p + U_xi_sigmoid*V_xi_tanh
            results[:, i, :] = p
        return results

class Model(torch.nn.Module):   # 这里的Module的forward是__forward__这样的，里面还有__call__这个东西，别忘记了
    def __init__(self, embedding_layer, emb_dim, label_nums=4):
        super().__init__()
        self.embeddings = embedding_layer
        self.gru = gru(emb_dim)
        self.classify = torch.nn.Linear(emb_dim, label_nums)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, batch_tokens_id, batch_labels = None):
        emb = self.embeddings(batch_tokens_id)  # emb : batch_size * seq_len * emb_dim
        gru_out = self.gru(emb) # gru_out : batch_size * seq_len * emb_dim
        predicts = torch.mean(gru_out, dim=1)   # 降维, 把dim=1这个维度降下来，变成一句话
        cls = self.classify(predicts)   # 分类，cls : batch_size * labels_num
        if batch_labels is not None:
            loss = self.loss_fn(cls, batch_labels)
            return loss
        else:
            return cls

def run_valid():
    model.eval()
    current_nums = 0
    with torch.no_grad():
        for batch_tokens_id, batch_labels in tqdm(testLoader, desc = "testing ..."):
            cls = model(batch_tokens_id)    # cls : batch_size * label_nums
            predict_label = torch.argmax(cls, dim=1)    # 每一行的预测
            current_nums += int(torch.sum(predict_label == batch_labels))

        acc = current_nums / len(testset)
        print(f"acc : {acc}")

if __name__ == "__main__":
    train_file = "data/train2.txt"
    test_file = "data/test2.txt"
    deepseek_model = AutoModel.from_pretrained("model/DeepSeek-R1-Distill-Qwen-1.5B")
    deepseek_tokenizer = AutoTokenizer.from_pretrained("model/DeepSeek-R1-Distill-Qwen-1.5B")
    deepseek_embeddings = deepseek_model.get_input_embeddings() # 返回是数据类型是torch.nn.embeddings
    emb_dim = deepseek_model.embed_tokens.weight.data.shape[1]
    seq_len = 32

    train_texts, train_labels = read_data(train_file)
    test_texts, test_labels = read_data(test_file)

    train_labels = mapping_label(train_labels)
    test_labels  = mapping_label(test_labels)

    trainset = MyDataset(train_texts, train_labels, 640)
    testset = MyDataset(test_texts, test_labels, 64)

    batch_size = 32
    trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    model = Model(deepseek_embeddings, emb_dim)

    epochs = 10
    lr = 0.001
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    for e in range(epochs):
        whole_loss = 0
        for batch_tokens_id, batch_labels in tqdm(trainLoader, desc="training ..."):
            loss = model(batch_tokens_id, batch_labels)
            whole_loss += loss
            loss.backward()
            optim.step()
            optim.zero_grad()
        print(f"loss : {whole_loss/batch_size}")
        run_valid()

    pass
