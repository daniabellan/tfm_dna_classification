import torch
import numpy as np

from src.dataset.domain.kmer_utils import generate_kmer_dict

class SequenceSignalsCollator:
    """
    A collator for processing sequence signals and k-mer sequences in a batch.

    This class:
    - Applies padding to k-mer sequences to ensure uniform length.
    - Applies zero-padding or truncates signals to a fixed maximum length.
    - Converts the processed data into PyTorch tensors for use in a DataLoader.

    Attributes:
        padding_idx (int): The index used for padding k-mer sequences.
        max_signal_length (int): The fixed length to which signals will be padded or truncated.
    """

    def __init__(self, 
                 kmers_size: int,
                 max_signal_length: int,
                 max_kmers_length: int):
        """
        Initializes the collator.

        Args:
            kmers_size (int): The size of k-mers used to determine padding index.
            max_signal_length (int, optional): The maximum allowed signal length.
                                               Signals longer than this will be truncated,
                                               and shorter ones will be zero-padded.
                                               Default is 1000.
        """
        self.padding_idx = len(generate_kmer_dict(kmers_size))  # Padding index for k-mers
        self.max_signal_length = max_signal_length  # Fixed signal length for CNNs
        self.max_kmers_length = max_kmers_length # Fixed kmers length

    def __call__(self, batch):
        """
        Processes a batch of sequence signals, applying padding and conversion to tensors.

        Args:
            batch (list of dicts): Each dict contains:
                - "signal_pa" (np.ndarray)
                - "full_processed_signal" (np.ndarray)
                - "window_signal" (np.ndarray)
                - "kmers" (np.ndarray)
                - "label" (int)

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "signal_tensor" (torch.Tensor): Padded signals (batch_size, max_signal_length).
                - "window_signal_tensor" (torch.Tensor): Windowed signals (batch_size, num_windows, window_size).
                - "padded_kmers" (torch.Tensor): Padded k-mer sequences (batch_size, max_kmer_len).
                - "labels" (torch.Tensor): Labels (batch_size).
        """
        # Extract each key separately
        signals = [item["window_signal"] for item in batch]
        kmer_sequences = [item["kmers"] for item in batch]
        labels = [item["label"] for item in batch]

        # Pad or truncate signals
        signal_tensor = self.pad_signals(signals)

        # Pad k-mer sequences
        padded_kmers = self.pad_kmers(kmer_sequences)

        # Convert to tensors
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return signal_tensor, padded_kmers, labels_tensor


    def pad_signals(self, signals: list) -> torch.Tensor:
        """
        Pads or truncates signals to a fixed length.

        Args:
            signals (list of np.ndarray): List of raw signals.

        Returns:
            torch.Tensor: Padded signals with shape (batch_size, max_signal_length).
        """
        padded_signals = []
        cut_signals = 0
        max_signal_length = 0
        for signal in signals:
            # Fill signals with zeros until every window have the same maximum length
            padded_signal = np.array(signal) 
            if len(padded_signal) < self.max_signal_length:
                # Fill with zeros until the desired length
                padding_needed = self.max_signal_length - len(padded_signal)
                padded_signal = np.pad(padded_signal, ((0, padding_needed), (0, 0)), mode='constant')
            elif len(padded_signal) > self.max_signal_length:
                # Trim signal if its needed
                padded_signal = padded_signal[:self.max_signal_length]
                cut_signals += 1
            padded_signals.append(padded_signal)

            if len(signal) > max_signal_length:
                max_signal_length = len(signal)

        padded_signals = np.array(padded_signals, dtype=np.float32)

        return torch.tensor(padded_signals, dtype=torch.float32)

    def pad_kmers(self, kmer_sequences: list) -> torch.Tensor:
        """
        Pads or truncates k-mer sequences to a fixed maximum length.

        Args:
            kmer_sequences (list of np.ndarray): List of k-mer index sequences.

        Returns:
            torch.Tensor: Padded k-mer sequences with shape (batch_size, max_kmers_length).
        """
        batch_size = len(kmer_sequences)
        num_truncated = 0  # Counter for sequences that exceed max_kmers_length

        # Initialize tensor with the padding index
        padded_kmers = np.full((batch_size, self.max_kmers_length), fill_value=self.padding_idx, dtype=np.int64)

        for i, kmers in enumerate(kmer_sequences):
            length = len(kmers)

            if length > self.max_kmers_length:
                kmers = kmers[:self.max_kmers_length]  # Truncate sequence
                num_truncated += 1
            else:
                # Pad sequence with the padding index if it's shorter
                kmers = np.pad(kmers, (0, self.max_kmers_length - length), constant_values=self.padding_idx)

            padded_kmers[i, :] = kmers  # Assign processed sequence

        return torch.tensor(padded_kmers, dtype=torch.long)
