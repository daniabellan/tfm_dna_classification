import numpy as np
from dataclasses import dataclass

@dataclass
class SignalSequenceData:
    signal_pa: np.ndarray
    sequence: str
    label: int
    full_processed_signal: np.ndarray = None
    window_signal: np.ndarray = None
    kmers: np.ndarray = None