import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple


class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = k.shape[-1]
        attentideon_score = torch.matmul(q,k.transpose(-2,-1))
        attention_score = attention_score /math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0,-1e9)
        attention_prob = F.softmax(attention_score, dim = -1)
        output = torch.matmul(attention_prob, v)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_batch = Q.size(0)
        
        def transform(x, layer): # 다중 헤드 어텐션에서 여러 헤드로 분할된 텐서가 병렬로 어텐션을 계산할 수 있도록
            out = layer(x)
            out = out.view(n_batch, -1, self.n_heads, self.d_model // self.n_heads) # view(n_batch, seq_len, n_heads, d_k)을 먼저 하여 텐서를 4차원으로 만들고
            out = out.transpose(1, 2) # transpose(1, 2)로 시퀀스 차원(seq_len)과 헤드 차원(n_heads)을 바꿔서 최종적으로 (batch_size, n_heads, seq_len, d_k) 형태로 
            return out 

        Q = transform(Q, self.query_layers)
        K = transform(K, self.key_layers)
        V = transform(V, self.value_layers)

        out = self.attention(Q, K, V, mask)  # Attention output: (n_batch, n_heads, seq_len, d_k)
        out = out.transpose(1, 2)  # (n_batch, seq_len, n_heads, d_k)
        out = out.contiguous().view(n_batch, -1, self.n_heads * (self.d_model // self.n_heads))  # Flatten to (n_batch, seq_len, d_model)
        out = self.fc(out)  # Apply final linear layer to project back to (n_batch, seq_len, d_model)
        return out

        
