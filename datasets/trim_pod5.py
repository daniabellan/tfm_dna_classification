import pod5
import pysam
import h5py
import numpy as np
from pathlib import Path


def process_pod5_files(file: Path):
    pod5_files = list(Path(pod5_data_path).rglob('*.pod5'))

    for file in pod5_files:
        print(f"Procesando archivo POD5: {file}")

        with pod5.Reader(file) as reader:
            for read in reader:
                read_id = str(read.read_id)

                read_trimmed = read.to_read()
                read_trimmed.signal = read_trimmed.signal[:100]

                writer = pod5.Writer("outputs/output.pod5")
                writer.add_read(read_trimmed)
                writer.close()
                pass



if __name__ == '__main__':
    pod5_data_path = "data/ecoli_k12/pod5_1"
    
    process_pod5_files(pod5_data_path)