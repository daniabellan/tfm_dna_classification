import torch

import numpy as np

class SequenceSignalsCollator:
    def __init__(self, padding_idx, max_signal_length=1000):
        self.padding_idx = padding_idx

        # Se añade esta variable para "fijar" el número de canales de la capa convolucional
        self.max_signal_length = max_signal_length

    def __call__(self, batch):
        signals, kmer_sequences, labels = zip(*batch)

        # Encontrar la longitud máxima de secuencia en el batch
        max_kmer_len = max(len(seq) for seq in kmer_sequences)
        
        # Aplicar padding a los k-mers
        padded_kmer_sequences = [
            seq + [self.padding_idx] * (max_kmer_len - len(seq)) for seq in kmer_sequences
        ]

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
        padded_kmer_sequences = torch.tensor(padded_kmer_sequences, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return signal_tensor, padded_kmer_sequences, labels_tensor
