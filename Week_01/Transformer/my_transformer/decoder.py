import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNormalization(d_model=d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.enc_dec_attention = MultiHeadAttention(d_model, n_heads)
        self.norm2 = LayerNormalization(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = FeedForwardLayer(d_model, d_ff=d_ff)
        self.norm3 = LayerNormalization(d_model)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. compute self attention
        _x = x
        x = self.self_attention(q=x, k=x, v=x, mask=tgt_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if memory is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=memory, v=memory, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
