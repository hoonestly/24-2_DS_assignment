import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()
        
        # Positional Encoding 계산
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        
        # 텐서를 register_buffer로 저장
        self.register_buffer('encoding', encoding.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        # 입력 시퀀스 길이 가져오기
        _, seq_len, _ = x.size()
        
        # 입력에 positional encoding 추가
        pos_embed = self.encoding[:, :seq_len, :]
        return x + pos_embed
