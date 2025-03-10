import numpy as np
from dataclasses import dataclass
import h5py
import time

from src.dataset.domain.dataclass import SignalSequenceData
from src.utils.logging_config import logger

def load_h5_file(dataset_path:str,
                 num_samples:int,
                 class_idx:int,
                 rng:np.random.Generator):
    
    start = time.time()

    h5_file = []
    with h5py.File(dataset_path, "r") as f:
        sampled_reads = rng.choice(list(f.keys()), 
                                   size = num_samples, 
                                   replace = False,
                                   shuffle=True)
        
        for read_id in sampled_reads:
            # Access to sequence
            sequence = f[f"{read_id}/sequence"]
            # If sequence is stored as string, decode
            if isinstance(sequence, h5py._hl.dataset.Dataset):
                sequence = sequence[()].decode()  # Bytes to string
            
            # Access to electric signal (stored as np.array)
            signal_pa = f[f"{read_id}/signal_pa"][()]  
            
            # Create data object
            data = SignalSequenceData(signal_pa=signal_pa,
                                      sequence=sequence,
                                      label=class_idx)

            h5_file.append(data)

    print(f"** HDF5 file loaded successfully! \n"
        f"   Path: {dataset_path}\n"
        f"   Time elapsed: {(time.time() - start):.4f} seconds\n"
        f"   Samples loaded: {num_samples}")

    logger.info(f"** HDF5 file loaded successfully! \n"
        f"   Path: {dataset_path}\n"
        f"   Time elapsed: {(time.time() - start):.4f} seconds\n"
        f"   Samples loaded: {num_samples}")

    return h5_file
