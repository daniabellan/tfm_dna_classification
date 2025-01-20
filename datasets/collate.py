import torch
import numpy as np

class SequenceSignalsCollator:
    def __init__(self, vocab, padding_idx):
        self.vocab = vocab
        self.padding_idx = padding_idx

    def __call__(self, batch):
        signals, sequences, labels = zip(*batch)

        # Encontrar la longitud máxima de secuencia en el batch
        max_len = max(len(seq) for seq in sequences)
        
        # Codificar las secuencias y hacer padding
        sequences_idx = []
        for seq in sequences:
            seq_idx = [self.vocab[base] for base in seq]
            # Aplicar padding a las secuencias para igualar las longitudes
            padded_seq = seq_idx + [self.padding_idx] * (max_len - len(seq))
            sequences_idx.append(padded_seq)

        # Padding a las señales
        padded_signals = []
        max_segments = max([len(sample[0]) for sample in batch])
        for signal in signals:
            # Rellenar señales con ceros para que todas tengan el mismo número de segmentos
            while len(signal) < max_segments:
                signal.append(np.zeros(len(signal[0]), dtype=np.float32))
            
            padded_signals.append(np.array(signal, dtype=np.float32))


        signal_tensor = torch.tensor(np.array(padded_signals), dtype=torch.float32)
        sequences_tensor = torch.tensor(sequences_idx, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return signal_tensor, sequences_tensor, labels_tensor
