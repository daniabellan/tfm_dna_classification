import torch
import torch.nn as nn
import mlflow
import psutil

class PositionalEncoding(nn.Module):
    """
    Implements Positional Encoding using nn.Embedding.

    Args:
        max_len (int): Maximum sequence length.
        d_model (int): Model embedding dimension.
    """
    def __init__(self, max_len: int, d_model: int):
        super(PositionalEncoding, self).__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.position_embedding(self.positions[:, :seq_len])
        return x + pos_enc
