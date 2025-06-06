import re
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, AutoModel, AutoConfig


def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.read().split("\n")

    pattern = re.compile(r"^(.*?)\s+(\d+)$")
    dataset = []
    for line in data:
        matches = pattern.match(line)
        if matches:
            text, label = matches.groups()
            dataset.append([text, int(label)])

    return dataset[:5000]

class MyDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx][0]
        label = self.dataset[idx][1]

        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=seq_len, return_tensors='pt')
        input_ids = inputs["input_ids"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "label": torch.tensor(label, dtype=torch.long)
        }

# def My_collate_fn(batch):
#     batch_input_ids = [item["input_ids"] for item in batch]
#     batch_labels = [item["label"] for item in batch]
#     new_batch_input_ids = []
#     for item in batch_input_ids:
#         # 长了就截断
#         if len(item) > seq_len:
#             new_batch_input_ids.append(item[:seq_len])
#         else:   # 短了就补齐
#             new_input_ids = item + [tokenizer.vocab["[PAD]"]] * (seq_len-len(item))
#             new_batch_input_ids.append(new_input_ids)

#     return torch.tensor(new_batch_input_ids, dtype=torch.long), torch.tensor(batch_labels, dtype=torch.long)

class MyModel(torch.nn.Module):
    def __init__(self, bge_model, labels_num, hidden_size=768):
        super().__init__()
        self.emb = bge_model
        self.cls = torch.nn.Linear(hidden_size, labels_num)

    def forward(self, input_ids):
        bge_output = self.emb(input_ids, attention_mask=(input_ids!=0)).last_hidden_state
        attention_mask = (input_ids != 0).unsqueeze(-1).float()
        masked_output = bge_output * attention_mask
        sum_output = masked_output.sum(dim=1)
        lengths = attention_mask.sum(dim=1)
        sentence_embeddings = sum_output / lengths

        logits = self.cls(sentence_embeddings)

        return logits
    
class CustomSentenceClassifier(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(config._name_or_path)
        self.classifier = torch.nn.Linear(self.backbone.config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = attention_mask.unsqueeze(-1).float()
        masked_output = last_hidden_state * attention_mask
        sum_output = masked_output.sum(dim=1)
        lengths = attention_mask.sum(dim=1)
        sentence_embeddings = sum_output / lengths

        logits = self.classifier(sentence_embeddings)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
    
def get_accuracy():
    model.eval()
    current_nums = 0
    total_nums = len(testset)
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc = "testing ... "):
            logits = model(batch["input_ids"])
            pre_labels = torch.argmax(logits, dim=-1)
            current_nums += (pre_labels == batch["label"]).sum().item()

    accuracy = current_nums / total_nums
    return accuracy

if __name__ == "__main__":
    data_file = "day30RAG/src2(VLLM)/NER/data/all_dataset.txt"
    pre_model_path = "models/bge-base-zh-v1.5"
    dataset = read_data(data_file)

    tokenizer = AutoTokenizer.from_pretrained(pre_model_path)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
    bge_model = AutoModel.from_pretrained(pre_model_path)

    batch_size = 64
    seq_len = 20
    epochs = 10
    lr = 1e-4
    label_nums = 5

    train_dataset = MyDataset(trainset, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_dataset = MyDataset(testset, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)


    model = MyModel(bge_model, label_nums)
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        for batch in tqdm(train_dataloader, desc = "training ... "):
            probilities = model(batch["input_ids"])
            # pre_labels = probilities[:, 0, :]
            loss = loss_fn(probilities, batch["label"])

            loss.backward()
            optim.step()
            optim.zero_grad()

        print(f"e : {e},loss: {loss}")
        
    print("acc: ", get_accuracy())


    save_path = "day30RAG/src2_VLLM/NER/checkpoint-hf"

    # 保存 config（可以使用原模型的 config 添加自定义字段）
    config = AutoConfig.from_pretrained(pre_model_path)
    config.num_labels = label_nums
    model.config = config

    # 保存模型和 tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # torch.save(model, 'day30RAG/src2(VLLM)/NER/model/model.pth')
    