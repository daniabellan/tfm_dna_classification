import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNKmerBranch(nn.Module):
    """
    CNN-based sequence processing using multiple convolutional layers.

    Args:
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        num_filters (int): Number of CNN filters.
        kernel_sizes (list): List of kernel sizes.
    """
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes=[3, 5, 7]):
        super(CNNKmerBranch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k//2) for k in kernel_sizes])
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), embed_dim)

    def forward(self, sequences):
        x = self.embedding(sequences).permute(0, 2, 1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = torch.cat([self.global_pool(conv).squeeze(2) for conv in x], dim=1)
        return self.fc(x)

class BiLSTMKmerBranch(nn.Module):
    """
    BiLSTM-based sequence processing.

    Args:
        vocab_size (int): Vocabulary size.
        hidden_dim (int): LSTM hidden dimension.
        embed_dim (int): Embedding dimension.
        num_layers (int): Number of LSTM layers.
    """
    def __init__(self, vocab_size, hidden_dim, embed_dim=128, num_layers=2):
        super(BiLSTMKmerBranch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, sequences):
        x = self.embedding(sequences)
        x, _ = self.lstm(x)
        x = self.global_pool(x.permute(0, 2, 1)).squeeze(2)
        return self.fc(x)
