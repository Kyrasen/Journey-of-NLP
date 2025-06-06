from inputEncoding1 import Encoding_embedding
from positionalEncoding2 import Positional_Encoding
from multiHeadAttention3 import MultiHeadAttention
from layerNorm4 import LayerNorm
from feedForward5 import FeedForward
import torch

class Encoder(torch.nn.Module):
    def __init__(self, num_heads, d_model, vocab_size):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.emb = Encoding_embedding(vocab_size, d_model)
        self.positional_encoding = Positional_Encoding()
        self.multiHeadAttention = MultiHeadAttention(self.num_heads, d_model)
        self.layerNorm = LayerNorm(self.d_model)
        self.feedForward = FeedForward(self.d_model)

    def forward(self, input_ids):
        inputs_emb = self.emb(input_ids)
        pe_output = self.positional_encoding(inputs_emb)
        mha_output = self.multiHeadAttention(pe_output,pe_output,pe_output )    # ?
        lan_output = self.layerNorm(mha_output + pe_output)
        ff_output = self.feedForward(lan_output)

        output = self.layerNorm(ff_output + lan_output)

        return output

if __name__ == "__main__":
    
    d_model = 512
    batch_size = 16
    vocab_size = 100
    seq_len = 200
    num_heads = 8

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    encoder = Encoder(num_heads, d_model)
    encoder_output = encoder(input_ids)

    pass