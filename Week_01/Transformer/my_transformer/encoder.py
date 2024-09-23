import torch.nn as nn
from torch import Tensor
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForwardLayer(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        attn_out = self.self_attn(query=x, key=x, value=x, mask=mask)
        attn_out = self.dropout1(attn_out)
        out = self.residual1(x, attn_out)
        out = self.norm1(out)
        
        ff_out = self.ff(out)
        ff_out = self.dropout2(ff_out)
        out = self.residual2(out, ff_out)
        out = self.norm2(out)
        
        return out