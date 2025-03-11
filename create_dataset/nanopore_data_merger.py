import pod5
from ont_fast5_api.fast5_interface import get_fast5_file
import pysam
import h5py
import numpy as np
import os
import argparse
from pathlib import Path

def read_sam_file(sam_file_path: str, 
                  max_allowed_reads:int=100_000,
                  n_trim_reads:int=40_000,
                  verbose: bool = True) -> dict:
    """
    Reads a SAM file and extracts read sequences into a dictionary.
    
    :param sam_file_path: Path to the SAM file.
    :param verbose: If True, prints process details.
    :return: Dictionary with read IDs as keys and sequences as values.
    """
    if verbose:
        print(f"Opening SAM file: {sam_file_path}")
    
    samfile = pysam.AlignmentFile(sam_file_path, "r", check_sq=False)
    reads_dict = {read.query_name: {"sequence": read.query_sequence, "signal_pa": None} 
                  for read in samfile.fetch(until_eof=True)}
    
    total_reads = len(reads_dict)

    if verbose:
        print(f"Total reads in SAM file: {total_reads}")

    # If the number of reads is larger than X, trim to random reads
    # This avoid RAM limitation
    if total_reads > max_allowed_reads:
        rng = np.random.default_rng(seed=42)  
        selected_keys = rng.choice(list(reads_dict.keys()), size=n_trim_reads, replace=False) 
        
        reads_dict = {key: reads_dict[key] for key in selected_keys}  

        if verbose:
            print(f"Selected {n_trim_reads} random reads from {total_reads}")


    return reads_dict

def process_pod5_file(file: Path, reads_dict: dict, verbose: bool = True) -> int:
    """
    Processes a POD5 file and updates the reads dictionary with signal data.
    
    :param file: Path to the POD5 file.
    :param reads_dict: Dictionary containing read sequences.
    :param verbose: If True, prints process details.
    :return: Number of reads not found in the POD5 file.
    """
    not_detected = 0
    detected = 0
    
    if verbose:
        print(f"Processing POD5 file: {file}")
    
    with pod5.Reader(file) as reader:
        for read in reader:
            read_id = str(read.read_id)
            
            if read_id in reads_dict:
                reads_dict[read_id]["signal_pa"] = read.signal_pa
                detected += 1
            else:
                not_detected += 1
    
    if verbose:
        print(f"Processed reads from {file}: {detected}")
    
    return not_detected

def process_fast5_file(file: Path, reads_dict: dict, verbose: bool = True) -> int:
    """
    Processes a Fast5 file and updates the reads dictionary with signal data.
    
    :param file: Path to the Fast5 file.
    :param reads_dict: Dictionary containing read sequences.
    :param verbose: If True, prints process details.
    :return: Number of reads not found in the Fast5 file.
    """
    
    if verbose:
        print(f"Processing Fast5 file: {file}")

    with get_fast5_file(file, mode='r') as f5:
        f5_reads = f5.get_read_ids()

        # Conversion to set to quick search
        reads_keys_set = set(reads_dict)
        
        # For loop only with detected reads
        detected_values = [value for value in f5_reads if value in reads_keys_set]
        
        not_detected = len(f5_reads) - len(detected_values)

        for read_idx, read_id in enumerate(detected_values):
            if verbose and read_idx % 100 == 0:  
                print(f"Processing read {read_idx}/{len(detected_values)}")

            if read_id in reads_keys_set:
                read = f5.get_read(read_id)
                raw_data = read.get_raw_data()
                metadata = read.get_channel_info()

                offset, range_scaling, digitisation = metadata["offset"], metadata["range"], metadata["digitisation"]

                reads_dict[read_id]["signal_pa"] = (raw_data + offset) * range_scaling / digitisation

    if verbose:
        print(f"Processed reads from {file}: {len(detected_values)} detected, {not_detected} not found.")

    return not_detected

def match_reads(nanopore_data_path: str, reads_dict: dict, verbose: bool = True) -> dict:
    """
    Matches reads from POD5 and Fast5 files to those in the reads dictionary.
    
    :param nanopore_data_path: Directory containing Fast5 or POD5 files.
    :param reads_dict: Dictionary containing read sequences.
    :param verbose: If True, prints process details.
    :return: Filtered dictionary with reads that contain both sequence and signal data.
    """
    pod5_files = list(Path(nanopore_data_path).rglob('*.pod5'))
    fast5_files = list(Path(nanopore_data_path).rglob('*.fast5'))
    
    not_detected = 0
    
    if verbose:
        print(f"Found {len(pod5_files)} POD5 files and {len(fast5_files)} Fast5 files in directory: {nanopore_data_path}")
    
    for file in fast5_files:
        not_detected += process_fast5_file(file, reads_dict, verbose)
    
    for file in pod5_files:
        not_detected += process_pod5_file(file, reads_dict, verbose)
    
    filtered_dict = {read_id: data for read_id, data in reads_dict.items()
                      if data.get("sequence") and data.get("signal_pa") is not None}
    
    if verbose:
        print(f"Reads not detected in POD5/Fast5 files: {not_detected}")
        print(f"Total valid reads with both sequence and signal: {len(filtered_dict)}")
    
    return filtered_dict

def save_dict_h5(dictionary: dict, file_path: str, verbose: bool = True):
    """
    Saves the processed reads dictionary into an HDF5 file.
    
    :param dictionary: Processed reads dictionary.
    :param file_path: Path to the output HDF5 file.
    :param verbose: If True, prints process details.
    """
    if verbose:
        print(f"Saving dictionary to {file_path}...")
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with h5py.File(file_path, 'w') as f:
        for read_id, data in dictionary.items():
            f.create_dataset(f"{read_id}/sequence", data=data["sequence"])
            f.create_dataset(f"{read_id}/signal_pa", data=np.array(data["signal_pa"], dtype=np.float32))
    
    if verbose:
        print(f"Dictionary successfully saved to {file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process SAM and Fast5/POD5 files.")
    parser.add_argument('sam_file_path', type=str, help="Path to the SAM file")
    parser.add_argument('nanopore_data_path', type=str, help="Directory containing nanopore data files (.fast5 or .pod5)")
    parser.add_argument('output_h5_path', type=str, help="Path to the output HDF5 file")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose mode")
    
    args = parser.parse_args()
    
    reads_dict = read_sam_file(args.sam_file_path, args.verbose)
    filtered_dict = match_reads(args.nanopore_data_path, reads_dict, args.verbose)
    save_dict_h5(filtered_dict, args.output_h5_path, args.verbose)
