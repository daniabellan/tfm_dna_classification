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

