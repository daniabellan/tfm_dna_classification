import pod5
from ont_fast5_api.fast5_interface import get_fast5_file
import pysam
import h5py
import numpy as np
from pathlib import Path


def read_sam_file(sam_file_path: str, verbose: bool = False) -> dict:
    """
    Lee un archivo SAM/BAM y extrae las secuencias de los reads en un diccionario.

    Args:
        sam_file_path (str): Ruta al archivo SAM/BAM.
        verbose (bool): Si es True, imprime mensajes de progreso.

    Returns:
        dict: Diccionario con 'read_id' como clave y diccionario con 'sequence' y 'signal_pa' como valores.
    """
    if verbose:
        print(f"Abriendo archivo SAM: {sam_file_path}")

    samfile = pysam.AlignmentFile(sam_file_path, "r", check_sq=False)
    
    reads_dict = {
        read.query_name: {
            "sequence": read.query_sequence,
            "signal_pa": None  # Campo vacío para la señal eléctrica
        }
        for read in samfile.fetch(until_eof=True)
    }

    if verbose:
        print(f"Total de reads en el archivo SAM: {len(reads_dict)}")

    return reads_dict


def process_pod5_file(file: Path, reads_dict: dict, verbose: bool = False) -> int:
    """
    Procesa un archivo POD5 y actualiza el diccionario con las señales correspondientes.

    Args:
        file (Path): Ruta al archivo POD5.
        reads_dict (dict): Diccionario con la información de los reads.
        verbose (bool): Si es True, imprime mensajes de progreso.

    Returns:
        int: Número de reads no detectados en el archivo POD5.
    """
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
            # Raw data en int16
            raw_data = read.get_raw_data()

            
            # Scale offset
            metadata = read.get_channel_info()
            offset = metadata["offset"]
            range_scaling = metadata["range"]
            digitisation = metadata["digitisation"]

            # Convert signal to pA (picoAmpers)
            raw_signals_pa = (raw_data + offset) * range_scaling / digitisation
            
            # Add signal to dictionary
            reads_dict[read_id]["signal_pa"] = raw_signals_pa
            detected += 1
        
        else:
            not_detected += 1

    if verbose:
        print(f"Reads procesados en {file}: {detected}")
    
    return not_detected


def match_reads(nanopore_data_path: str, reads_dict: dict, verbose: bool = False) -> dict:
    """
    Procesa archivos POD5 para actualizar el campo 'signal_pa' en reads_dict y retorna
    un nuevo diccionario con los reads que contienen tanto 'sequence' como 'signal_pa'.

    Args:
        nanopore_data_path (str): Ruta al directorio que contiene archivos .pod5 o fast5.
        reads_dict (dict): Diccionario con información de reads.
        verbose (bool): Si es True, imprime mensajes de progreso.

    Returns:
        dict: Nuevo diccionario con solo los reads que contienen 'sequence' y 'signal_pa'.
    """
    

    pod5_files = list(Path(nanopore_data_path).rglob('*.pod5'))
    fast5_files = list(Path(nanopore_data_path).rglob('*.fast5'))

    not_detected = 0
    if verbose:
        print(f"Se encontraron {len(pod5_files)} archivos POD5 en el directorio: {nanopore_data_path}")
        print(f"Se encontraron {len(fast5_files)} archivos FAST5 en el directorio: {nanopore_data_path}")

    
    # Procesamos los archivos y contamos los reads no detectados 
    if len(fast5_files) > 0:
        for file in fast5_files:
            not_detected += process_fast5_file(file, reads_dict, verbose)

    if len(pod5_files) > 0:
        for file in pod5_files:
            not_detected += process_pod5_file(file, reads_dict, verbose)

    # Filtrar los reads que tienen tanto 'sequence' como 'signal_pa'
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
    """
    Guarda el diccionario en un archivo HDF5.

    Args:
        dictionary (dict): Diccionario a guardar.
        file_path (str): Ruta al archivo HDF5 donde se guardarán los datos.
        verbose (bool): Si es True, imprime mensajes de progreso.
    """
    if verbose:
        print(f"Guardando diccionario en {file_path}...")

    with h5py.File(file_path, 'w') as f:
        for read_id, data in dictionary.items():
            # Guardar la secuencia y la señal como datasets separados
            f.create_dataset(f"{read_id}/sequence", data=data["sequence"])
            f.create_dataset(f"{read_id}/signal_pa", data=np.array(data["signal_pa"], dtype=np.float32))

    if verbose:
        print(f"Diccionario guardado exitosamente en {file_path}")


if __name__ == '__main__':
    sam_file_path = "data/mm39/reads.sam"
    nanopore_data_path = "data/mm39/fast5"
    
    # Establecer verbose en True para mostrar más detalles
    verbose = True
    
    # Leer el archivo SAM para obtener el diccionario de reads
    reads_dict = read_sam_file(sam_file_path, verbose)
    
    # Procesar los archivos POD5 y actualizar el diccionario con las señales
    filtered_dict = match_reads(nanopore_data_path, reads_dict, verbose)
    
    # Guardar el diccionario resultante en un archivo HDF5 con el mismo nombre que el directorio POD5
    h5_file_path = nanopore_data_path.rstrip('/') + '.h5'  # Crear el nombre del archivo HDF5
    save_dict_h5(filtered_dict, h5_file_path, verbose)
