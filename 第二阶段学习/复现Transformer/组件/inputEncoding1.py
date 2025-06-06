import random
import torch
import math

class Encoding_embedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.d_model = d_model


    def forward(self, input_ids):
        embs = self.embedding(input_ids)

        return (embs * math.sqrt(self.d_model))

# if __name__ == "__main__":
#     vocab_size = 100
#     d_model = 512
#     batch_size = 16
#     seq_len = 100

#     input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
#     encoding_emb = Encoding_embedding(vocab_size, d_model)

#     input_embedding = encoding_emb(input_ids)
#     print(f"input_embedding's shape: {input_embedding.shape}")

