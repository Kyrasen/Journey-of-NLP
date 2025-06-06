import torch

class FeedForward(torch.nn.Module):
    def __init__(self, d_model,dff = 2048):
        super().__init__()
        self.W1 = torch.nn.Linear(d_model, dff)
        self.W2 = torch.nn.Linear(dff, d_model)
        self.relu = torch.nn.ReLU()

    def forward(self, input_emb):
        x = self.W1(input_emb)
        x = self.relu(x)
        x = self.W2(x)
        return torch.dropout(x, 0.1, train=False)


# if __name__ == "__main__":
#     d_model = 512
#     batch_size = 16
#     vocab_size = 100
#     seq_len = 200
#     num_heads = 8

#     input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
#     emb = torch.nn.Embedding(vocab_size, d_model)
#     input_emb = emb(input_ids)

#     ff = FeedForward()
#     ff_output = ff(input_emb)

#     pass