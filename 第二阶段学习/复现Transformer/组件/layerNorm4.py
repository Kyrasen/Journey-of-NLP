'''
    层归一化
'''
import torch

class LayerNorm(torch.nn.Module):
    def __init__(self, d_model,eps = 1e-5):
        super().__init__()
        self.eps = 1e-5

        self.gamma = torch.nn.Parameter(torch.ones(d_model))    # 类似于斜率
        self.beta = torch.nn.Parameter(torch.zeros(d_model))    # 类似于偏置常数

    def forward(self, inputs_emb):
        mean = torch.mean(inputs_emb, dim=-1, keepdim=True)
        std = torch.std(inputs_emb, dim=-1, keepdim=True)
        output = ((inputs_emb - mean) / (std+self.eps)) * self.gamma + self.beta

        return output

# if __name__ == "__main__":
#     d_model = 512
#     batch_size = 16
#     vocab_size = 100
#     seq_len = 200
#     num_heads = 8

#     input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
#     emb = torch.nn.Embedding(vocab_size, d_model)
#     input_emb = emb(input_ids)

#     lan_block = LayerNorm()
#     output = lan_block(input_emb)

#     pass

