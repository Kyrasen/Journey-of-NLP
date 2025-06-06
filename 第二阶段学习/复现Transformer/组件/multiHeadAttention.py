import torch

def scaled_dot_product_attention(Q, V, K):
    '''
        d_k = d_model / heads
        input -> (batch_size, heads, seq_len, d_k)
        Q, V, K: shape: (batch_size, heads, seq_len, d_k)
    '''
    d_k = Q.shape[-1]
    attention_scores = (Q @ K.transpose(-2,-1)) / torch.sqrt(d_k)
    attention_weights = torch.softmax(attention_scores) @ V

    return attention_weights

if __name__ == "__main__":
    batch_size = 16
    heads = 8
    seq_len = 100
    d_model = 128
    d_k = int( d_model / heads)
    

    pass