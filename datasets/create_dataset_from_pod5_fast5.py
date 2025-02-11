import pod5
from ont_fast5_api.fast5_interface import get_fast5_file
import pysam
import h5py
import numpy as np
import os
import argparse
from pathlib import Path


def read_sam_file(sam_file_path: str, verbose: bool = False) -> dict:
    if verbose:
        print(f"Abriendo archivo SAM: {sam_file_path}")

    samfile = pysam.AlignmentFile(sam_file_path, "r", check_sq=False)
    
    reads_dict = {
        read.query_name: {
            "sequence": read.query_sequence,
            "signal_pa": None
        }
        for read in samfile.fetch(until_eof=True)
    }

    if verbose:
        print(f"Total de reads en el archivo SAM: {len(reads_dict)}")

    return reads_dict


def process_pod5_file(file: Path, reads_dict: dict, verbose: bool = False) -> int:
    not_detected = 0
    detected = 0
    if verbose:
        print(f"Procesando archivo POD5: {file}")

    with pod5.Reader(file) as reader:
        for read in reader:
            read_id = str(read.read_id)
            
            if read_id in reads_dict:
                reads_dict[read_id]["signal_pa"] = read.signal_pa
                detected += 1
            else:
                not_detected += 1
    
    if verbose:
        print(f"Reads procesados en {file}: {detected}")
    
    return not_detected


def process_fast5_file(file: Path, reads_dict: dict, verbose: bool = False) -> int:
    not_detected = 0
    detected = 0
    if verbose:
        print(f"Procesando archivo Fast5: {file}")
    
    f5 = get_fast5_file(file, mode='r')
    
    file_reads = f5.get_read_ids()
    for read_idx, read_id in enumerate(file_reads):
        read = f5.get_read(read_id)

        if read_id in reads_dict:
            raw_data = read.get_raw_data()
            metadata = read.get_channel_info()
            offset = metadata["offset"]
            range_scaling = metadata["range"]
            digitisation = metadata["digitisation"]

            raw_signals_pa = (raw_data + offset) * range_scaling / digitisation
            
            reads_dict[read_id]["signal_pa"] = raw_signals_pa
            detected += 1
        
        else:
            not_detected += 1

    if verbose:
        print(f"Reads procesados en {file}: {detected}")
    
    return not_detected


def match_reads(nanopore_data_path: str, reads_dict: dict, verbose: bool = False) -> dict:
    pod5_files = list(Path(nanopore_data_path).rglob('*.pod5'))
    fast5_files = list(Path(nanopore_data_path).rglob('*.fast5'))

    not_detected = 0
    if verbose:
        print(f"Se encontraron {len(pod5_files)} archivos POD5 en el directorio: {nanopore_data_path}")
        print(f"Se encontraron {len(fast5_files)} archivos FAST5 en el directorio: {nanopore_data_path}")

    if len(fast5_files) > 0:
        for file in fast5_files:
            not_detected += process_fast5_file(file, reads_dict, verbose)

    if len(pod5_files) > 0:
        for file in pod5_files:
            not_detected += process_pod5_file(file, reads_dict, verbose)

    filtered_dict = {
        read_id: data
        for read_id, data in reads_dict.items()
        if data.get("sequence") and data.get("signal_pa") is not None
    }

    if verbose:
        print(f"Reads no detectados en archivos POD5: {not_detected}")
        print(f"Total de reads válidos con 'sequence' y 'signal_pa': {len(filtered_dict)}")

    return filtered_dict


def save_dict_h5(dictionary, file_path, verbose: bool = False):
    if verbose:
        print(f"Guardando diccionario en {file_path}...")

    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with h5py.File(file_path, 'w') as f:
        for read_id, data in dictionary.items():
            f.create_dataset(f"{read_id}/sequence", data=data["sequence"])
            f.create_dataset(f"{read_id}/signal_pa", data=np.array(data["signal_pa"], dtype=np.float32))

    if verbose:
        print(f"Diccionario guardado exitosamente en {file_path}")


if __name__ == '__main__':
    # Configurar argparse para aceptar los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Procesar archivos SAM y FAST5/POD5.")
    parser.add_argument('sam_file_path', type=str, help="Ruta al archivo SAM")
    parser.add_argument('nanopore_data_path', type=str, help="Ruta al directorio de datos nanopore (.fast5 o .pod5)")
    parser.add_argument('output_h5_path', type=str, help="Ruta al archivo HDF5 de salida")
    parser.add_argument('--verbose', action='store_true', help="Mostrar detalles del proceso")

    # Parsear los argumentos
    args = parser.parse_args()

    # Leer el archivo SAM para obtener el diccionario de reads
    reads_dict = read_sam_file(args.sam_file_path, args.verbose)
    
    # Procesar los archivos POD5 y actualizar el diccionario con las señales
    filtered_dict = match_reads(args.nanopore_data_path, reads_dict, args.verbose)
    
    # Guardar el diccionario resultante en el archivo HDF5
    save_dict_h5(filtered_dict, args.output_h5_path, args.verbose)
