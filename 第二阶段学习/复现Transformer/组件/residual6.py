import torch
from layerNorm4 import LayerNorm

class Residual(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lan = LayerNorm(d_model)


    def forward(self, x, sublayer):
        res = x + sublayer(x)
        return self.lan(res)