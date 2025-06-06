from inputEncoding1 import Encoding_embedding
from positionalEncoding2 import Positional_Encoding
from multiHeadAttention3 import MultiHeadAttention
from layerNorm4 import LayerNorm
from feedForward5 import FeedForward
import torch
from encoder import Encoder

class Decoder(torch.nn.Module):
    def __init__(self, num_heads, d_model, output_vocab_size):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.emb = Encoding_embedding(vocab_size, d_model)
        self.positional_encoding = Positional_Encoding()
        self.multiHeadAttention = MultiHeadAttention(self.num_heads, d_model)
        self.layerNorm = LayerNorm(self.d_model)
        self.feedForward = FeedForward(self.d_model)

        self.linear = torch.nn.Linear(d_model, output_vocab_size)

    def forward(self, output_ids, encoder_output):
        outputs_emb = self.emb(output_ids)
        pe_output = self.positional_encoding(outputs_emb)
        mha_output1 = self.multiHeadAttention(pe_output,pe_output,pe_output )    # ?
        lan_output1 = self.layerNorm(mha_output1 + pe_output)

        mha_output2 = self.multiHeadAttention(encoder_output, encoder_output, lan_output1)
        lan_output2 = self.layerNorm(mha_output2 + lan_output1)

        ff_output = self.feedForward(lan_output2)
        lan_output3 = self.layerNorm(ff_output + lan_output2)
    
        output = self.linear(lan_output3)


        return torch.softmax(output, dim=-1)

if __name__ == "__main__":
    
    d_model = 512
    batch_size = 16
    vocab_size = 100
    seq_len = 200
    num_heads = 8
    output_vocab_size = 150

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    encoder = Encoder(num_heads, d_model, vocab_size)
    encoder_output = encoder(input_ids)

    output_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    decoder = Decoder(num_heads, d_model,output_vocab_size)
    decoder_output = decoder(output_ids,encoder_output)

    pass


