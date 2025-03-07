import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residual_block import ResidualBlock  
from src.models.domain.utils import PositionalEncoding

class SignalBranch(nn.Module):
    """
    Signal processing branch using CNNs, Residual Blocks, and Transformers.

    Args:
        input_channels (int): Number of input channels.
        max_len (int): Maximum sequence length for positional encoding.
        transformer_layers (int): Number of Transformer Encoder layers.
        transformer_dropout (float): Dropout rate for the Transformer.
    """
    def __init__(self, input_channels, max_len=1500, transformer_layers=2, transformer_dropout=0.5):
        super(SignalBranch, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, stride=3, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dilation=1),
            ResidualBlock(128, 256, stride=2, dilation=2),
            ResidualBlock(256, 512, stride=2, dilation=4),
            ResidualBlock(512, 1024, stride=2, dilation=8)
        )
        self.global_pool = nn.AdaptiveMaxPool1d(8)
        self.positional_encoding = PositionalEncoding(max_len=max_len, d_model=1024)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024, nhead=8, dim_feedforward=2048, dropout=transformer_dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def forward(self, signals):
        x = F.relu(self.conv1(signals))
        x = self.pool1(x)
        x = self.residual_blocks(x)
        x = self.global_pool(x)        
        x = x.permute(0, 2, 1)  # Reshape for transformer
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)         
        return F.normalize(x, p=2, dim=1)
