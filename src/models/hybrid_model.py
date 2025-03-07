# from typing import Literal
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.residual_block import ResidualBlock  

# from src.dataset.domain.kmer_utils import generate_kmer_dict

# ###########################################
# #            UTILIDADES                   #
# ###########################################

# class PositionalEncoding(nn.Module):
#     def __init__(self, max_len: int, d_model: int):
#         super(PositionalEncoding, self).__init__()
#         self.position_embedding = nn.Embedding(max_len, d_model)
#         self.register_buffer("positions", torch.arange(max_len).unsqueeze(0))

#     def forward(self, x):
#         seq_len = x.size(1)
#         pos_enc = self.position_embedding(self.positions[:, :seq_len])
#         return x + pos_enc

# ###########################################
# #         RAMA DE SEÑALES (SIGNALS)        #
# ###########################################

# class SignalBranch(nn.Module):
#     def __init__(self, 
#                  input_channels, 
#                  max_len=1500, 
#                  transformer_layers=2,
#                  transformer_dropout=0.5):
        
#         super(SignalBranch, self).__init__()
#         self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, stride=3, padding=2)
#         self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
#         self.layer1 = ResidualBlock(64, 128, stride=2, dilation=1)
#         self.layer2 = ResidualBlock(128, 256, stride=2, dilation=2)
#         self.layer3 = ResidualBlock(256, 512, stride=2, dilation=4)
#         self.layer4 = ResidualBlock(512, 1024, stride=2, dilation=8)
#         self.global_pool = nn.AdaptiveMaxPool1d(8)
#         self.positional_encoding = PositionalEncoding(max_len=max_len, d_model=1024)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=1024,
#             nhead=8,
#             dim_feedforward=2048,
#             dropout=transformer_dropout,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

#     def forward(self, signals):
#         x = F.relu(self.conv1(signals))
#         x = self.pool1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.global_pool(x)        
#         x = x.permute(0, 2, 1)            
#         x = self.positional_encoding(x)
#         x = self.transformer(x)
#         x = torch.mean(x, dim=1)         
#         x = F.normalize(x, p=2, dim=1)
#         return x

# ###########################################
# #       RAMA DE SECUENCIAS (SEQUENCES)      #
# ###########################################

# class CNNKmerBranch(nn.Module):
#     def __init__(self, 
#                  vocab_size, 
#                  embed_dim, 
#                  num_filters, 
#                  kernel_sizes=[3, 5, 7]):
        
#         super(CNNKmerBranch, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)
#         self.convs = nn.ModuleList([
#             nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k//2) for k in kernel_sizes
#         ])
#         self.global_pool = nn.AdaptiveMaxPool1d(1)
#         self.fc = nn.Linear(num_filters * len(kernel_sizes), embed_dim)

#     def forward(self, sequences):
#         x = self.embedding(sequences)         # (batch, seq_len, embed_dim)
#         x = x.permute(0, 2, 1)                  # (batch, embed_dim, seq_len)
#         conv_outputs = [F.relu(conv(x)) for conv in self.convs]
#         pooled_outputs = [self.global_pool(conv).squeeze(2) for conv in conv_outputs]
#         x = torch.cat(pooled_outputs, dim=1)
#         x = self.fc(x)
#         return x

# class BiLSTMKmerBranch(nn.Module):
#     def __init__(self, vocab_size, hidden_dim, embed_dim=128, num_layers=2):
#         super(BiLSTMKmerBranch, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
#                             batch_first=True, bidirectional=True, dropout=0.3)
#         self.global_pool = nn.AdaptiveMaxPool1d(1)
#         self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

#     def forward(self, sequences):
#         x = self.embedding(sequences)   # (batch, seq_len, embed_dim)
#         x, _ = self.lstm(x)             # (batch, seq_len, hidden_dim * 2)
#         x = x.permute(0, 2, 1).contiguous()
#         x = self.global_pool(x).squeeze(2)
#         x = self.fc(x)
#         return x

# ###########################################
# #         CLASIFICADORES (HEADS)          #
# ###########################################

# class CombinedClassifier(nn.Module):
#     def __init__(self, signals_dim, sequences_dim, num_classes):
#         super(CombinedClassifier, self).__init__()
#         self.bn = nn.BatchNorm1d(signals_dim + sequences_dim)
#         self.fc1 = nn.Linear(signals_dim + sequences_dim, 2048)
#         self.fc2 = nn.Linear(2048, 512)
#         self.fc3 = nn.Linear(512, num_classes)
#         self.dropout1 = nn.Dropout(0.6)
#         self.dropout2 = nn.Dropout(0.6)

#     def forward(self, x):
#         x = self.bn(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return x

# class SignalsClassifier(nn.Module):
#     def __init__(self, signals_dim, num_classes):
#         super(SignalsClassifier, self).__init__()
#         self.fc1 = nn.Linear(signals_dim, 2048)
#         self.fc2 = nn.Linear(2048, 512)
#         self.fc3 = nn.Linear(512, num_classes)
#         self.dropout1 = nn.Dropout(0.6)
#         self.dropout2 = nn.Dropout(0.6)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return x

# class SequencesClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(SequencesClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 2048)
#         self.fc2 = nn.Linear(2048, 512)
#         self.fc3 = nn.Linear(512, num_classes)
#         self.dropout1 = nn.Dropout(0.6)
#         self.dropout2 = nn.Dropout(0.6)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return x

# ###########################################
# #     MODELO HIBRIDO COMPLETO           #
# ###########################################

# class HybridSequenceClassifier(nn.Module):
#     def __init__(self, 
#                  input_channels,
#                  kmers_size:int, 
#                  num_classes:int, 
#                  use_signals:bool, 
#                  use_sequences:bool,
#                  sequence_branch=Literal['cnn', 'bilstm'],  # 'cnn' para CNNKmerBranch o 'bilstm' para BiLSTMKmerBranch
#                  embed_dim:int=16, 
#                  max_len=1500,
#                  hidden_dim=256,
#                  num_filters=256,
#                  kernel_sizes=[3, 5, 7]):
#         """
#         Modelo híbrido que combina procesamiento de señales eléctricas y secuencias.
#         El parámetro 'sequence_branch' permite seleccionar la implementación de la rama de secuencias.
#         """
#         super(HybridSequenceClassifier, self).__init__()
#         self.use_signals = use_signals
#         self.use_sequences = use_sequences
#         self.sequence_branch = sequence_branch

#         vocab_size = len(generate_kmer_dict(kmers_size=kmers_size))+1
        

#         if use_signals:
#             self.signal_branch = SignalBranch(input_channels, max_len=max_len)
        
#         if use_sequences:
#             if sequence_branch == 'cnn':
#                 self.sequence_branch = CNNKmerBranch(vocab_size=vocab_size, 
#                                                      embed_dim=embed_dim, 
#                                                      num_filters=num_filters, 
#                                                      kernel_sizes=kernel_sizes)
#                 seq_out_dim = embed_dim  # Dimensión de salida del CNNKmerBranch
#             elif sequence_branch == 'bilstm':
#                 self.sequence_branch = BiLSTMKmerBranch(vocab_size=vocab_size, 
#                                                         hidden_dim=hidden_dim, 
#                                                         embed_dim=embed_dim)
#                 seq_out_dim = hidden_dim  # Dimensión de salida del BiLSTMKmerBranch
#             else:
#                 raise ValueError("sequence_branch debe ser 'cnn' o 'bilstm'.")

#         # Seleccionar el clasificador según las ramas utilizadas
#         if use_signals and use_sequences:
#             self.classifier = CombinedClassifier(signals_dim=1024, sequences_dim=seq_out_dim, num_classes=num_classes)
#         elif use_signals:
#             self.classifier = SignalsClassifier(signals_dim=1024, num_classes=num_classes)
#         elif use_sequences:
#             self.classifier = SequencesClassifier(input_dim=seq_out_dim, num_classes=num_classes)

#     def forward(self, signals, sequences):
#         outputs = []
#         if self.use_signals:
#             x_signals = self.signal_branch(signals)
#             outputs.append(x_signals)
#         if self.use_sequences:
#             x_sequences = self.sequence_branch(sequences)
#             outputs.append(x_sequences)
        
#         if len(outputs) > 1:
#             x = torch.cat(outputs, dim=1)
#         else:
#             x = outputs[0]
        
#         x = self.classifier(x)
#         return x

from typing import Literal
import torch
import torch.nn as nn
from src.models.signal_branch import SignalBranch
from src.models.sequence_branch import CNNKmerBranch, BiLSTMKmerBranch
from src.models.classifiers import CombinedClassifier, SequencesClassifier, SignalsClassifier
from src.dataset.domain.kmer_utils import generate_kmer_dict

class HybridSequenceClassifier(nn.Module):
    """
    Hybrid model combining signal and sequence processing.

    Args:
        input_channels (int): Input channels for signal processing.
        kmers_size (int): Size of k-mers.
        num_classes (int): Number of output classes.
        use_signals (bool): Whether to use signal processing branch.
        use_sequences (bool): Whether to use sequence processing branch.
        sequence_branch (str): 'cnn' or 'bilstm' sequence branch type.
    """
    def __init__(self, 
                 input_channels, 
                 kmers_size, 
                 num_classes, 
                 sequence_branch:Literal['cnn', 'bilstm'],
                 use_signals=True, 
                 use_sequences=True):
        super(HybridSequenceClassifier, self).__init__()
        
        vocab_size = len(generate_kmer_dict(kmers_size)) + 1
        self.use_signals = use_signals
        self.use_sequences = use_sequences

        # Initialize signal processing branch (if enabled)
        if use_signals:
            self.signal_branch = SignalBranch(input_channels)
            signals_dim = 1024  # Fixed dimension from SignalBranch

        # Initialize sequence processing branch (if enabled)
        if use_sequences:
            if sequence_branch == 'cnn':
                self.sequence_branch = CNNKmerBranch(vocab_size, 16, 256)
                sequences_dim = 16  # Output of CNNKmerBranch
            elif sequence_branch == 'bilstm':
                self.sequence_branch = BiLSTMKmerBranch(vocab_size, 256)
                sequences_dim = 256  # Output of BiLSTMKmerBranch
            else:
                raise ValueError("sequence_branch must be 'cnn' or 'bilstm'.")

        # Select appropriate classifier based on branch configuration
        if use_signals and use_sequences:
            self.classifier = CombinedClassifier(signals_dim, sequences_dim, num_classes)
        elif use_signals:
            self.classifier = SignalsClassifier(signals_dim, num_classes)
        elif use_sequences:
            self.classifier = SequencesClassifier(sequences_dim, num_classes)
        else:
            raise ValueError("At least one branch (signals or sequences) must be enabled.")

    def forward(self, signals, sequences):
        outputs = []
        if self.use_signals:
            outputs.append(self.signal_branch(signals))
        if self.use_sequences:
            outputs.append(self.sequence_branch(sequences))
        
        x = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
        return self.classifier(x)

