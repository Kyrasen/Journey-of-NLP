import torch
import math

class Positional_Encoding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_emb):
        batch_size, seq_len, d_model = input_emb.shape
        pos = torch.arange(0, seq_len).unsqueeze(1)
        even = torch.arange(0, d_model, 2)
        exp_term = torch.exp((-even * math.log(10000)) / d_model)

        term = pos * exp_term
        res = torch.zeros(size=(batch_size, seq_len, d_model))

        res[:, :, 0::2] = torch.sin(term)
        res[:, :, 1::2] = torch.cos(term)


        return torch.dropout(input_emb + res, 0.1, train=False)


# if __name__ == "__main__":
#     vocab_size = 100
#     d_model = 512
#     batch_size = 16
#     seq_len = 200
#     emb = torch.nn.Embedding(vocab_size, d_model)

#     input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
#     input_emb = emb(input_ids)

#     pe = Positional_Encoding(input_emb)
#     pe_output = pe()

#     pass