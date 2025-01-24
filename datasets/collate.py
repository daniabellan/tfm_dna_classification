import torch
import numpy as np

class SequenceSignalsCollator:
    def __init__(self, vocab, padding_idx, max_signal_length=1000):
        self.vocab = vocab
        self.padding_idx = padding_idx

        # Se añade esta variable para "fijar" el número de canales de la capa convolucional
        self.max_signal_length = max_signal_length

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
        cut_signals = 0
        max_signal_length = 0
        for signal in signals:
            # Rellenar señales con ceros para que todas tengan la longitud máxima deseada
            padded_signal = np.array(signal)  # Convertir la señal a numpy si es necesario
            if len(padded_signal) < self.max_signal_length:
                # Rellenar con ceros hasta la longitud deseada
                padding_needed = self.max_signal_length - len(padded_signal)
                padded_signal = np.pad(padded_signal, ((0, padding_needed), (0, 0)), mode='constant')
            elif len(padded_signal) > self.max_signal_length:
                # Si la señal es más larga de lo que queremos, la cortamos (opcional)
                padded_signal = padded_signal[:self.max_signal_length]
                cut_signals += 1
            padded_signals.append(padded_signal)

            if len(signal) > max_signal_length:
                max_signal_length = len(signal)

        # print(f"Max signal_length: {max_signal_length}")
        signal_tensor = torch.tensor(np.array(padded_signals), dtype=torch.float32)
        sequences_tensor = torch.tensor(sequences_idx, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return signal_tensor, sequences_tensor, labels_tensor
