from torch.utils.data import Dataset, DataLoader
import numpy as np

def encode(sentence, max_len=15):
    sentence = sentence[:max_len]
    wordIdx = [vocab[word] for word in sentence]
    wordIdx += [vocab["PAD"]] * (max_len - len(sentence))
    sentence_emb = onehot_emb[wordIdx]
    return sentence_emb
        


class Draft(Dataset):
    def __init__(self, corpus, labels):
        self.corpus = corpus
        self.labels = labels

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        sentence_emb = encode(self.corpus[idx])
        return sentence_emb, self.labels[idx]
    
def read_data(file):
    corpus = []
    labels = []
    with open(file, "r", encoding = "utf-8") as f:
        lines = f.read().split("\n")

        for line in lines:
            if line:
                try:
                    sentence, label = line.split(" ")
                except:
                    pass
                try:
                    sentence, label = line.split("  ")
                except:
                    pass
                try:
                    sentence, label = line.split("\t")
                except:
                    pass
            corpus.append(sentence)
            labels.append(int(label))

    return corpus, labels

def tokenize(corpus):
    vocab = {"PAD":0}
    for sentence in corpus:
        for word in sentence:
            vocab.setdefault(word, len(vocab))

    return vocab

def onehot(labels, classes = 2):
    res = np.zeros((len(labels), classes))
    rows = np.arange(len(labels))
    res[rows, labels] = 1
    return res

def softmax(output):
    exp_val = np.exp(output)
    sum_val = np.sum(exp_val, axis = 1, keepdims = True)

    return exp_val/sum_val

if __name__ == "__main__":

    # file, corpus, labels, onehot_emp都是全局变量，因为它们是在if __name__ == "__main__":这个函数的代码块的顶端定义的，
    file = "data/text.txt"
    corpus, labels = read_data(file)

    vocab = tokenize(corpus)
    word_num = len(vocab)

    onehot_emb = np.eye(word_num)
    
    trainset = Draft(corpus, onehot(labels))
    trainloader = DataLoader(trainset, batch_size = 1, shuffle=True)

    in_features = len(vocab)
    out_features = classes = 2

    W = np.random.normal(size = (in_features, out_features))

    epochs = 20
    lr = 0.01
    batch_size = 1

    for epoch in range(epochs):
        for batch_inputs, batch_labels in trainloader:
            batch_inputs = batch_inputs.numpy()
            batch_labels = batch_labels.numpy()

            output = batch_inputs @ W
            predicts = softmax(output)
            loss = -np.sum(batch_inputs*np.log(predicts)) / len(batch_inputs)

            G = (predicts - batch_labels)/len(batch_inputs)
   



    pass


