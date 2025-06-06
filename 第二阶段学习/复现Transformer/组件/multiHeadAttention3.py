import torch
import math

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads,d_model, dropout = 0.1):
        super().__init__()
        self.softmax = torch.nn.Softmax()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model/self.num_heads
        self.q_proj_weights = torch.nn.Linear(self.d_model, self.d_model)
        self.v_proj_weights = torch.nn.Linear(self.d_model, self.d_model)
        self.k_proj_weights = torch.nn.Linear(self.d_model, self.d_model)

        self.o_proj_weights = torch.nn.Linear(self.d_model, self.d_model)

    def attention(self, Q, K, V):
        term = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        atten_scores = torch.softmax(term , dim=-1) # 这里的torch.softmax是一个函数
        atten_scores = atten_scores @ V
        
        return atten_scores
    
    def forward(self, query, key, value):
        # (batch, seq_len, d_model)
        batch_size, seq_len, _ = query.shape
        d_k = int(self.d_model / self.num_heads)
        Q = self.q_proj_weights(query)
        K = self.k_proj_weights(key)
        V = self.v_proj_weights(value)

        Q = Q.reshape(batch_size, self.num_heads, seq_len, d_k)
        K = K.reshape(batch_size, self.num_heads, seq_len, d_k)
        V = V.reshape(batch_size, self.num_heads, seq_len, d_k)

        atten_scores = self.attention(Q, K, V)
        output = self.o_proj_weights(atten_scores.reshape(batch_size, seq_len, self.d_model))
        
        return torch.dropout(output, 0.1, train=False)

# if __name__ == "__main__":
#     d_model = 512
#     batch_size = 16
#     vocab_size = 100
#     seq_len = 200
#     num_heads = 8

#     input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
#     emb = torch.nn.Embedding(vocab_size, d_model)
#     input_emb = emb(input_ids)

#     mha_block = MultiHeadAttention(input_emb, num_heads)
#     mha_block(input_emb)

